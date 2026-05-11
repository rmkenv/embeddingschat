# 🛰️ AEF Explorer

**AlphaEarth Foundations Embeddings → Ollama Cloud · No GEE Required**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/aef-explorer/blob/main/AEF_Explorer_Colab.ipynb)
[![CI](https://github.com/YOUR_USERNAME/aef-explorer/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/aef-explorer/actions)

> **Attribution**: *"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."*  
> License: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## What This Is

A Streamlit app + Python pipeline for exploring [AlphaEarth Foundations (AEF)](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/) 64-dimensional satellite embeddings — **without Google Earth Engine**.

Data is read directly from [Source Cooperative COGs](https://source.coop/tge-labs/aef) via HTTP windowed reads (only your AOI pixels, not the full 8192×8192 tile). A local [Ollama](https://ollama.com) model interprets the embedding statistics.

### What you get
- **PCA false-color** visualization (PC1/2/3 → RGB from 64-dim space)
- **Channel activation** bar chart (which AEF dimensions dominate your AOI)
- **Change detection** (cosine distance between two years, per-pixel)
- **Ollama interpretation** — landscape characterization, change analysis, or technical report
- **JSON export** of embedding summaries
- **PNG download** of PCA false-color

---

## Quickstart (local)

```bash
git clone https://github.com/YOUR_USERNAME/aef-explorer
cd aef-explorer

# Install deps (requires GDAL system libs — see below)
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama pull llama3.2
ollama serve

# Run app
streamlit run app.py
```

### System dependencies (GDAL/PROJ)

**macOS:**
```bash
brew install gdal proj
pip install rasterio --no-binary rasterio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libgdal-dev gdal-bin libproj-dev
pip install -r requirements.txt
```

**Windows:** Use [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or WSL2.

---

## Deploy to Streamlit Community Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select this repo → set **Main file: `app.py`**
4. Deploy

`packages.txt` handles system-level GDAL/PROJ installation automatically on Streamlit Cloud.

> **Note on Ollama**: Streamlit Community Cloud is serverless — it can't reach `localhost:11434`.  
> Point the Ollama host to a remote instance (e.g. a VPS, Fly.io, or Modal deployment running `ollama serve`),  
> or use the **"Skip Ollama"** checkbox to use the app purely for embedding extraction and visualization.

---

## Architecture

```
app.py                          ← Streamlit UI
aef_core.py                     ← Pipeline logic (no Streamlit dependency)
├── load_index()                → downloads tile index CSV from source.coop once
├── find_tiles_for_aoi()        → spatial filter (Shapely intersection)
├── build_tile_url()            → constructs HTTPS COG URL
├── read_tile_window()          → VSICURL windowed read, dequantize int8→float32
├── pca_rgb()                   → sklearn PCA → uint8 RGB
├── cosine_similarity_map()     → dot product to reference pixel
├── cosine_change_map()         → 1 − cosine_sim(t1, t2)
├── summarize_embeddings()      → structured dict for LLM
└── query_ollama()              → POST /api/chat to Ollama

.streamlit/config.toml          ← Dark theme config
requirements.txt                ← Python deps
packages.txt                    ← System deps (GDAL, PROJ) for Streamlit Cloud
.github/workflows/ci.yml        ← CI smoke tests
```

---

## Data Details

### Source
- **Source Cooperative**: `source.coop/tge-labs/aef` (community mirror, no auth)
- **GCS bucket**: `gs://alphaearth_foundations` (requester-pays, official)
- **GEE dataset**: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (original, requires GEE)

### Format
- Annual embeddings 2017–2024 (source.coop) / 2017–2025 (GCS)
- 64 channels (A00–A63), 10m resolution, global
- Files: 8192×8192 px COGs per UTM zone tile, int8 quantized

### Dequantization
```python
# int8 [-127..127] → float32 [-1..1]
float_val = ((raw_int8 / 127.5) ** 2) * np.sign(raw_int8)
# -128 = NoData (masked pixel → NaN)
```

### "Bottom-up" COG quirk
Source Cooperative tiles use positive y-resolution (bottom-left origin). rasterio handles this automatically for windowed reads, but GDAL CLI tools may require `--config GDAL_TIFF_OVR_BLOCKSIZE 512` or manual flip.

---

## Preset AOIs

| Preset | BBox (lon_min, lat_min, lon_max, lat_max) |
|--------|-------------------------------------------|
| Catonsville, MD | -76.755, 39.240, -76.680, 39.290 |
| Baltimore, MD   | -76.720, 39.250, -76.550, 39.380 |
| Washington DC   | -77.120, 38.800, -76.910, 38.990 |
| New York City   | -74.050, 40.680, -73.920, 40.800 |
| Chesapeake Bay  | -76.500, 38.700, -76.000, 39.100 |
| Atlanta, GA     | -84.500, 33.650, -84.300, 33.850 |

---

## Ollama Setup

### Option A — Ollama Cloud (recommended for Streamlit Community Cloud)

1. Get an API key at [ollama.com](https://ollama.com)
2. In the app sidebar: check **"Use Ollama Cloud API"** and paste your key
3. For Streamlit Community Cloud: add the secret in the app's Secrets dashboard:
   ```
   OLLAMA_API_KEY = "your-key-here"
   ```
   The app auto-loads it on startup — no manual entry needed.

Uses the OpenAI-compatible `/v1/chat/completions` endpoint with Bearer auth. Any model on Ollama Cloud works: `llama3.2`, `llama3.1`, `mistral`, `qwen2.5:14b`, etc.

### Option B — Local Ollama

```bash
ollama pull llama3.2
ollama serve   # listens on localhost:11434
```

Uncheck "Use Ollama Cloud API" in the sidebar. Works for local dev; not compatible with Streamlit Community Cloud (serverless).

### Local secrets setup

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Add your Ollama Cloud key — never commit this file
```

---

## CLI Usage (no Streamlit)

```bash
# Single year
python aef_run.py --preset catonsville --year 2023 --model llama3.2

# Change detection
python aef_run.py --preset baltimore --year 2019 --compare-year 2023 \
  --task change --save-pca out_pca.png --save-summary out.json

# Extract only, no LLM
python aef_run.py --preset dc --year 2022 --no-llm
```

---

## License

Code: MIT  
Data: CC-BY 4.0 — *"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."*
