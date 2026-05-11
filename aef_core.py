"""
aef_core.py
-----------
AlphaEarth Foundations (AEF) embedding pipeline via Source Cooperative COGs.
No GEE required. No Google auth required.

Source: https://source.coop/tge-labs/aef
License: CC-BY 4.0 — "The AlphaEarth Foundations Satellite Embedding dataset
         is produced by Google and Google DeepMind."
"""

import io
import json
import logging
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
import requests

# suppress rasterio/GDAL noise
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning if False else UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aef_pipeline")

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError as e:
    raise ImportError("rasterio required: pip install rasterio") from e

try:
    from pyproj import Transformer
    from shapely.geometry import box, mapping
    from shapely.ops import transform as shapely_transform
except ImportError as e:
    raise ImportError("pyproj + shapely required: pip install pyproj shapely") from e


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_COOP_BASE = "https://data.source.coop/tge-labs/aef/v1/annual"
INDEX_CSV_URL = f"{SOURCE_COOP_BASE}/aef_index.csv"
MANIFEST_URL = f"{SOURCE_COOP_BASE}/manifest.txt"
NUM_CHANNELS = 64
NODATA_INT = -128

# UTM zone letters (bands C–X, excluding I and O)
_UTM_LETTERS = "CDEFGHJKLMNPQRSTUVWX"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AOI:
    """Axis-aligned bounding box in WGS84 (EPSG:4326)."""
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    name: str = "AOI"

    def center(self):
        return (self.lon_min + self.lon_max) / 2, (self.lat_min + self.lat_max) / 2

    def to_shapely(self):
        return box(self.lon_min, self.lat_min, self.lon_max, self.lat_max)


@dataclass
class EmbeddingResult:
    """Output from extract_embeddings()."""
    embeddings: np.ndarray          # shape (H, W, 64), float32, unit-norm per pixel
    mask: np.ndarray                # shape (H, W), bool — True = valid
    year: int
    utm_zone: str
    tile_url: str
    aoi: AOI
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# UTM helpers
# ---------------------------------------------------------------------------

def wgs84_to_utm_zone(lon: float, lat: float) -> str:
    """Return UTM zone string like '18N' or '33S'."""
    zone_num = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"
    return f"{zone_num}{hemisphere}"


def utm_epsg(zone_str: str) -> int:
    """'18N' → 32618, '33S' → 32733."""
    num = int(re.match(r"(\d+)", zone_str).group(1))
    base = 32600 if zone_str.endswith("N") else 32700
    return base + num


# ---------------------------------------------------------------------------
# Tile index / discovery
# ---------------------------------------------------------------------------

_INDEX_CACHE: Optional[list[dict]] = None


def load_index(cache_path: str = "/tmp/aef_index.csv") -> list[dict]:
    """
    Download (once) and parse the AEF tile index CSV.
    Returns list of dicts with keys: filename, year, zone, bbox_wgs84.
    
    The full GeoParquet index is cleaner but requires geopandas.
    We parse the CSV which has columns: filename, minx, miny, maxx, maxy (WGS84).
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    if os.path.exists(cache_path):
        log.info("Loading cached AEF index from %s", cache_path)
        raw = open(cache_path).read()
    else:
        log.info("Downloading AEF tile index (~few MB)…")
        r = requests.get(INDEX_CSV_URL, timeout=60)
        r.raise_for_status()
        raw = r.text
        with open(cache_path, "w") as f:
            f.write(raw)
        log.info("Index cached to %s", cache_path)

    lines = raw.strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    records = []
    for line in lines[1:]:
        parts = line.split(",")
        rec = dict(zip(header, parts))
        records.append(rec)

    _INDEX_CACHE = records
    log.info("Loaded %d tile records from index", len(records))
    return records


def find_tiles_for_aoi(aoi: AOI, year: int, index: list[dict]) -> list[dict]:
    """
    Filter index to tiles that intersect the AOI for a given year.
    Index CSV columns expected: filename (path-like), minx, miny, maxx, maxy.
    """
    aoi_box = aoi.to_shapely()
    matched = []
    year_str = str(year)

    for rec in index:
        fname = rec.get("filename", "")
        # filter by year substring in path
        if year_str not in fname:
            continue
        try:
            minx = float(rec.get("minx", rec.get("left", 0)))
            miny = float(rec.get("miny", rec.get("bottom", 0)))
            maxx = float(rec.get("maxx", rec.get("right", 0)))
            maxy = float(rec.get("maxy", rec.get("top", 0)))
        except (ValueError, TypeError):
            continue

        tile_box = box(minx, miny, maxx, maxy)
        if aoi_box.intersects(tile_box):
            matched.append({**rec, "tile_box": tile_box})

    return matched


def build_tile_url(filename: str) -> str:
    """Convert index filename to Source Cooperative HTTPS URL."""
    # Strip any leading slash or 'v1/annual/' prefix if present
    fname = filename.lstrip("/")
    if fname.startswith("v1/annual/"):
        fname = fname[len("v1/annual/"):]
    return f"{SOURCE_COOP_BASE}/{fname}"


# ---------------------------------------------------------------------------
# COG reading + dequantization
# ---------------------------------------------------------------------------

def dequantize(raw: np.ndarray) -> np.ndarray:
    """
    AEF dequantization: int8 [-127..127] → float32 [-1..1].
    Formula: ((x / 127.5)^2) * sign(x)
    NODATA (-128) pixels are left as NaN.
    """
    out = raw.astype(np.float32)
    nodata_mask = raw == NODATA_INT
    out[nodata_mask] = np.nan
    valid = ~nodata_mask
    out[valid] = ((out[valid] / 127.5) ** 2) * np.sign(out[valid])
    return out


def read_tile_window(
    url: str,
    aoi: AOI,
    utm_epsg_code: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Open a COG via VSICURL, find the intersection window for the AOI,
    read all 64 channels, dequantize.

    Returns:
        data    – float32 array (H, W, 64)
        mask    – bool array (H, W), True = valid pixel
        info    – dict with CRS, transform, shape metadata
    """
    vsicurl = f"/vsicurl/{url}"

    # Reproject AOI bbox to tile's UTM CRS
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg_code}", always_xy=True)
    x0, y0 = transformer.transform(aoi.lon_min, aoi.lat_min)
    x1, y1 = transformer.transform(aoi.lon_max, aoi.lat_max)
    # ensure correct order
    left, right = min(x0, x1), max(x0, x1)
    bottom, top = min(y0, y1), max(y0, y1)

    with rasterio.open(vsicurl) as src:
        log.info("Opened COG: %s  CRS=%s  shape=%s", url.split("/")[-1], src.crs, src.shape)

        # The source.coop COGs are "bottom-up" — rasterio may handle this,
        # but we clamp the window to valid bounds just in case.
        window = window_from_bounds(left, bottom, right, top, src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        raw = src.read(window=window)  # (64, H, W), int8
        info = {
            "crs": str(src.crs),
            "native_shape": src.shape,
            "window_shape": (raw.shape[1], raw.shape[2]),
            "transform": src.window_transform(window),
            "tile_url": url,
        }

    # Dequantize: (64, H, W) int8 → (64, H, W) float32
    data_f = dequantize(raw)

    # Build valid mask — any channel with NaN means pixel is masked
    mask = ~np.isnan(data_f).any(axis=0)  # (H, W)

    # Transpose to (H, W, 64)
    data_f = np.transpose(data_f, (1, 2, 0))

    # Replace NaN with 0 in masked pixels so downstream math doesn't break
    data_f[~mask] = 0.0

    log.info("Read window shape: %s  valid pixels: %d/%d",
             data_f.shape[:2], mask.sum(), mask.size)
    return data_f, mask, info


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def cosine_similarity_map(embeddings: np.ndarray, ref_pixel: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel cosine similarity to a reference embedding vector.
    embeddings: (H, W, 64) unit-norm vectors
    ref_pixel:  (64,) unit-norm vector
    Returns (H, W) float32 in [-1, 1]
    """
    # AEF embeddings are already unit-norm; dot product = cosine similarity
    return np.einsum("hwc,c->hw", embeddings, ref_pixel).astype(np.float32)


def change_magnitude_map(emb_t1: np.ndarray, emb_t2: np.ndarray) -> np.ndarray:
    """
    L2 distance between two embedding arrays as a proxy for land-cover change.
    Returns (H, W) float32.
    """
    diff = emb_t1 - emb_t2
    return np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)


def cosine_change_map(emb_t1: np.ndarray, emb_t2: np.ndarray) -> np.ndarray:
    """
    1 - cosine_similarity(t1, t2) per pixel — angular change metric.
    Returns (H, W) float32 in [0, 2].
    """
    dot = np.einsum("hwc,hwc->hw", emb_t1, emb_t2)
    return (1.0 - dot).astype(np.float32)


def pca_rgb(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Project 64-dim embeddings to 3 PCA components, return uint8 RGB (H, W, 3).
    Only fits PCA on valid (unmasked) pixels.
    """
    from sklearn.decomposition import PCA

    H, W, C = embeddings.shape
    flat = embeddings.reshape(-1, C)  # (N, 64)
    valid_flat = flat[mask.reshape(-1)]

    pca = PCA(n_components=3)
    projected_valid = pca.fit_transform(valid_flat)  # (Nvalid, 3)

    # Build full projected array
    projected = np.zeros((H * W, 3), dtype=np.float32)
    projected[mask.reshape(-1)] = projected_valid

    projected = projected.reshape(H, W, 3)

    # Normalize each channel to [0, 255]
    rgb = np.zeros_like(projected, dtype=np.uint8)
    for i in range(3):
        ch = projected[:, :, i]
        ch_valid = ch[mask]
        if ch_valid.ptp() > 0:
            ch = (ch - ch_valid.min()) / ch_valid.ptp()
        rgb[:, :, i] = (ch * 255).clip(0, 255).astype(np.uint8)

    return rgb


def summarize_embeddings(
    embeddings: np.ndarray,
    mask: np.ndarray,
    aoi: AOI,
    year: int,
    change_map: Optional[np.ndarray] = None,
) -> dict:
    """
    Produce a structured summary dict suitable for LLM consumption.
    """
    valid = embeddings[mask]
    n_valid = mask.sum()
    n_total = mask.size

    # Channel-wise stats on valid pixels
    ch_mean = valid.mean(axis=0)      # (64,)
    ch_std  = valid.std(axis=0)       # (64,)
    top3_ch = np.argsort(np.abs(ch_mean))[-3:][::-1].tolist()

    summary = {
        "aoi_name": aoi.name,
        "year": year,
        "bbox_wgs84": [aoi.lon_min, aoi.lat_min, aoi.lon_max, aoi.lat_max],
        "pixel_count_total": int(n_total),
        "pixel_count_valid": int(n_valid),
        "coverage_pct": round(100 * n_valid / n_total, 1) if n_total else 0,
        "embedding_dims": NUM_CHANNELS,
        "channel_mean_norm": round(float(np.linalg.norm(ch_mean)), 4),
        "top3_active_channels": top3_ch,
        "top3_channel_means": [round(float(ch_mean[i]), 4) for i in top3_ch],
        "mean_intra_aoi_variance": round(float(ch_std.mean()), 4),
    }

    if change_map is not None:
        valid_change = change_map[mask]
        summary.update({
            "change_metric": "cosine_distance",
            "change_mean": round(float(valid_change.mean()), 4),
            "change_std": round(float(valid_change.std()), 4),
            "change_p90": round(float(np.percentile(valid_change, 90)), 4),
            "pct_high_change": round(float((valid_change > 0.1).mean() * 100), 1),
        })

    return summary


# ---------------------------------------------------------------------------
# Ollama integration
# ---------------------------------------------------------------------------

def query_ollama(
    summary: dict,
    change_summary: Optional[dict] = None,
    model: str = "llama3.2",
    host: str = "https://api.ollama.ai",
    api_key: Optional[str] = None,
    task: str = "interpret",
) -> str:
    """
    Send embedding summary to Ollama (Cloud or local) and return the response text.

    Ollama Cloud (default):
        host    = "https://api.ollama.ai"
        api_key = "<your Ollama Cloud key>"
        Uses the OpenAI-compatible /v1/chat/completions endpoint with Bearer auth.

    Local fallback (no api_key):
        host    = "http://localhost:11434"
        api_key = None
        Uses the native /api/chat endpoint.

    task options: 'interpret', 'change', 'report'
    """
    if task == "change" and change_summary:
        prompt = f"""You are a geospatial analyst interpreting satellite embedding change signals.

Year 1 embedding summary:
{json.dumps(summary, indent=2)}

Year 2 embedding summary (same AOI):
{json.dumps(change_summary, indent=2)}

Based on these AlphaEarth Foundations 64-dimensional embedding statistics, provide:
1. What the embedding shift suggests about land-cover or environmental change
2. Which channels show the most change and what that might indicate
3. Confidence level and caveats
4. Recommended follow-up analysis

Be specific and quantitative where possible. Do not hallucinate specific land types — reason from the statistics provided."""

    elif task == "report":
        prompt = f"""You are a remote sensing scientist writing a technical brief.

AlphaEarth Foundations embedding extraction summary:
{json.dumps(summary, indent=2)}

Write a concise technical paragraph (3-5 sentences) suitable for a methods section or monitoring report.
Include: AOI location, year, embedding dimensionality, data coverage, and what the dominant embedding signal pattern suggests about the landscape."""

    else:
        prompt = f"""You are a geospatial AI assistant. Below is a statistical summary of AlphaEarth Foundations
satellite embeddings (64-dimensional pixel-level vectors, 10m resolution) extracted for a study area.

Embedding summary:
{json.dumps(summary, indent=2)}

Please interpret:
1. What the embedding statistics suggest about the character of this landscape
2. Which embedding dimensions are most active and what that could indicate
3. Data quality notes based on coverage percentage
4. Suggested downstream analyses given this embedding profile"""

    messages = [{"role": "user", "content": prompt}]

    # ── Ollama Cloud: OpenAI-compatible endpoint ──────────────────────────────
    if api_key:
        url = f"{host.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Ollama Cloud HTTP {e.response.status_code}: {e.response.text[:300]}]"
        except Exception as e:
            return f"[Ollama Cloud error: {e}]"

    # ── Local Ollama: native /api/chat ────────────────────────────────────────
    else:
        url = f"{host.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except httpx.ConnectError:
            return f"[Ollama not reachable at {host} — run 'ollama serve' and ensure model '{model}' is pulled]"
        except Exception as e:
            return f"[Ollama local error: {e}]"


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    aoi: AOI,
    year: int,
    ollama_model: str = "llama3.2",
    ollama_host: str = "https://api.ollama.ai",
    ollama_api_key: Optional[str] = None,
    compare_year: Optional[int] = None,
    task: str = "interpret",
) -> dict:
    """
    Full pipeline:
      1. Load tile index from Source Cooperative
      2. Find intersecting tiles for AOI + year
      3. Read + dequantize embeddings
      4. Compute analysis layers
      5. Summarize and query Ollama
    
    Returns a results dict with all outputs.
    """
    results = {"aoi": aoi, "year": year}

    # 1. Index
    index = load_index()

    # 2. Find tiles
    tiles = find_tiles_for_aoi(aoi, year, index)
    if not tiles:
        log.warning("No tiles found for AOI '%s' in year %d. Try a larger AOI or different year.", aoi.name, year)
        # Fall back to direct URL construction from UTM zone
        lon_c, lat_c = aoi.center()
        zone = wgs84_to_utm_zone(lon_c, lat_c)
        log.info("Falling back to UTM zone %s for direct tile access", zone)
        tiles = [{"filename": f"v1/annual/{year}/{zone}/", "zone": zone}]

    log.info("Found %d candidate tile(s) for %s / %d", len(tiles), aoi.name, year)

    # Use first matching tile
    tile = tiles[0]
    fname = tile.get("filename", "")
    url = build_tile_url(fname) if fname and not fname.endswith("/") else None

    # Extract UTM zone from tile record or derive from AOI center
    zone_match = re.search(r"/(\d+[NS])/", fname)
    utm_zone = zone_match.group(1) if zone_match else wgs84_to_utm_zone(*aoi.center())
    epsg = utm_epsg(utm_zone)

    results["utm_zone"] = utm_zone
    results["epsg"] = epsg

    if url is None:
        log.error("Could not construct tile URL from index record: %s", tile)
        results["error"] = "Could not resolve tile URL"
        return results

    # 3. Read COG
    log.info("Reading COG for year %d…", year)
    embeddings, mask, info = read_tile_window(url, aoi, epsg)
    results["embeddings_t1"] = embeddings
    results["mask_t1"] = mask
    results["info_t1"] = info

    # 4. Analysis layers
    # PCA false-color
    if mask.sum() > 3:
        results["pca_rgb"] = pca_rgb(embeddings, mask)

    # Optional change detection
    change_map = None
    emb_t2 = None
    mask_t2 = None
    summary_t2 = None

    if compare_year is not None:
        tiles2 = find_tiles_for_aoi(aoi, compare_year, index)
        if tiles2:
            url2 = build_tile_url(tiles2[0].get("filename", ""))
            log.info("Reading COG for comparison year %d…", compare_year)
            emb_t2, mask_t2, info2 = read_tile_window(url2, aoi, epsg)
            results["embeddings_t2"] = emb_t2
            results["mask_t2"] = mask_t2
            results["info_t2"] = info2

            joint_mask = mask & mask_t2
            change_map = cosine_change_map(embeddings, emb_t2)
            results["change_map"] = change_map

            summary_t2 = summarize_embeddings(emb_t2, mask_t2, aoi, compare_year, change_map)
            results["summary_t2"] = summary_t2

    # 5. Summarize
    summary_t1 = summarize_embeddings(embeddings, mask, aoi, year, change_map)
    results["summary_t1"] = summary_t1

    # 6. Ollama
    log.info("Querying Ollama model '%s'…", ollama_model)
    llm_response = query_ollama(
        summary_t1,
        change_summary=summary_t2,
        model=ollama_model,
        host=ollama_host,
        api_key=ollama_api_key,
        task=task if compare_year is None else "change",
    )
    results["llm_response"] = llm_response

    return results
