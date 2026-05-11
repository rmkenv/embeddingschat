"""
AlphaEarth Foundations → Ollama Explorer
Streamlit app — no GEE required
Source: source.coop/tge-labs/aef (CC-BY 4.0)
Attribution: "The AlphaEarth Foundations Satellite Embedding dataset
              is produced by Google and Google DeepMind."
"""

import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEF Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0b0f1a;
    --surface:  #131929;
    --border:   #1e2d47;
    --accent:   #00d4aa;
    --accent2:  #3b82f6;
    --warn:     #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Monospace for labels/stats */
.mono { font-family: 'Space Mono', monospace; font-size: 0.78rem; }

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
}
.metric-sub {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 2px;
}

/* LLM output */
.llm-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 20px 24px;
    font-size: 0.9rem;
    line-height: 1.7;
    white-space: pre-wrap;
    margin-top: 12px;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
}
.app-subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: -4px;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-green { background: rgba(0,212,170,0.15); color: var(--accent); border: 1px solid rgba(0,212,170,0.3); }
.badge-blue  { background: rgba(59,130,246,0.15); color: var(--accent2); border: 1px solid rgba(59,130,246,0.3); }
.badge-warn  { background: rgba(245,158,11,0.15);  color: var(--warn);   border: 1px solid rgba(245,158,11,0.3); }

/* Section headers */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 20px 0 12px;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #000;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.88; }

div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
}

.stAlert { border-radius: 6px; }

/* Channel bar chart area */
.ch-grid {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-top: 8px;
}
.ch-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
}
.ch-label { width: 28px; color: var(--muted); text-align: right; }
.ch-bar-wrap { flex: 1; background: var(--border); border-radius: 2px; height: 8px; }
.ch-bar { height: 8px; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.ch-val { width: 44px; color: var(--text); text-align: right; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (heavy deps only when needed) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def _import_core():
    from aef_core import (
        AOI, load_index, find_tiles_for_aoi, build_tile_url,
        read_tile_window, cosine_change_map, pca_rgb,
        summarize_embeddings, query_ollama, utm_epsg, wgs84_to_utm_zone,
        cosine_similarity_map,
    )
    return dict(
        AOI=AOI, load_index=load_index,
        find_tiles_for_aoi=find_tiles_for_aoi,
        build_tile_url=build_tile_url,
        read_tile_window=read_tile_window,
        cosine_change_map=cosine_change_map,
        pca_rgb=pca_rgb,
        summarize_embeddings=summarize_embeddings,
        query_ollama=query_ollama,
        utm_epsg=utm_epsg,
        wgs84_to_utm_zone=wgs84_to_utm_zone,
        cosine_similarity_map=cosine_similarity_map,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _load_index():
    core = _import_core()
    return core["load_index"]()


# ── Preset AOIs ───────────────────────────────────────────────────────────────
PRESETS = {
    "Custom": None,
    "Catonsville, MD":    (-76.755, 39.240, -76.680, 39.290),
    "Baltimore, MD":      (-76.720, 39.250, -76.550, 39.380),
    "Washington DC":      (-77.120, 38.800, -76.910, 38.990),
    "New York City":      (-74.050, 40.680, -73.920, 40.800),
    "Chesapeake Bay":     (-76.500, 38.700, -76.000, 39.100),
    "Atlanta, GA":        (-84.500, 33.650, -84.300, 33.850),
    "Chicago, IL":        (-87.750, 41.750, -87.550, 41.950),
    "Houston, TX":        (-95.500, 29.650, -95.200, 29.850),
}

YEARS = list(range(2017, 2025))

TASKS = {
    "Interpret landscape":  "interpret",
    "Change detection":     "change",
    "Technical report":     "report",
}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style="font-size:2.4rem">🛰️</span>
  <div>
    <div class="app-title">AEF Explorer</div>
    <div class="app-subtitle">AlphaEarth Foundations Embeddings · Source Cooperative COGs · Local Ollama LLM · No GEE</div>
  </div>
</div>
<div style="margin-bottom:20px">
  <span class="badge badge-green">CC-BY 4.0</span>&nbsp;
  <span class="badge badge-blue">64-dim · 10m</span>&nbsp;
  <span class="badge badge-warn">2017–2024</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-head">Area of Interest</div>', unsafe_allow_html=True)

    preset = st.selectbox("Preset AOI", list(PRESETS.keys()), index=1)

    if preset == "Custom":
        col1, col2 = st.columns(2)
        lon_min = col1.number_input("Lon min", value=-76.755, format="%.4f")
        lat_min = col2.number_input("Lat min", value=39.240,  format="%.4f")
        lon_max = col1.number_input("Lon max", value=-76.680, format="%.4f")
        lat_max = col2.number_input("Lat max", value=39.290,  format="%.4f")
        aoi_name = st.text_input("Name", value="Custom AOI")
    else:
        bbox = PRESETS[preset]
        lon_min, lat_min, lon_max, lat_max = bbox
        aoi_name = preset

    st.markdown('<div class="section-head">Years</div>', unsafe_allow_html=True)
    year1 = st.selectbox("Primary year", YEARS, index=YEARS.index(2023))
    enable_change = st.checkbox("Enable change detection", value=False)
    year2 = None
    if enable_change:
        year2 = st.selectbox("Compare year", YEARS, index=YEARS.index(2019))
        if year2 == year1:
            st.warning("Select a different year for comparison.")
            year2 = None

    st.markdown('<div class="section-head">Ollama</div>', unsafe_allow_html=True)

    # Load API key from Streamlit secrets if available
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("OLLAMA_API_KEY", "")
    except Exception:
        pass

    use_cloud = st.checkbox("Use Ollama Cloud API", value=bool(_secret_key), help="Uncheck to use a local Ollama instance instead")

    if use_cloud:
        ollama_host = st.text_input("Ollama Cloud host", value="https://api.ollama.ai",
                                     help="Ollama Cloud base URL")
        ollama_api_key = st.text_input(
            "API key",
            value=_secret_key,
            type="password",
            help="Set OLLAMA_API_KEY in Streamlit secrets to avoid entering it here",
        )
        st.caption("Uses `/v1/chat/completions` (OpenAI-compatible) with Bearer auth.")
    else:
        ollama_host = st.text_input("Local Ollama host", value="http://localhost:11434")
        ollama_api_key = ""
        st.caption("Uses `/api/chat` — requires `ollama serve` running locally.")

    ollama_model = st.text_input("Model", value="llama3.2",
                                  help="e.g. llama3.2, llama3.1, mistral, qwen2.5:14b")
    task_label   = st.selectbox("LLM task", list(TASKS.keys()),
                                 index=1 if enable_change else 0)
    task = TASKS[task_label]

    st.markdown('<div class="section-head">Options</div>', unsafe_allow_html=True)
    skip_llm = st.checkbox("Skip Ollama (extract only)", value=False)

    st.markdown("---")
    run_btn = st.button("▶  Run Pipeline", use_container_width=True)

    st.markdown("""
    <div style="font-size:0.62rem; color:#475569; margin-top:16px; line-height:1.5">
    Data: <code>source.coop/tge-labs/aef</code><br>
    Attribution: <em>"The AlphaEarth Foundations Satellite Embedding dataset
    is produced by Google and Google DeepMind."</em>
    </div>
    """, unsafe_allow_html=True)


# ── Main area — idle state ────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="
        background: #131929;
        border: 1px dashed #1e2d47;
        border-radius: 10px;
        padding: 48px 32px;
        text-align: center;
        margin-top: 24px;
    ">
        <div style="font-size:3rem; margin-bottom:12px">🌍</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#e2e8f0; margin-bottom:8px">
            Configure your AOI and hit Run Pipeline
        </div>
        <div style="font-size:0.82rem; color:#64748b; max-width:480px; margin:0 auto; line-height:1.6">
            Reads 64-dimensional AlphaEarth Foundations embeddings directly from
            Source Cooperative Cloud-Optimized GeoTIFFs via HTTP range requests.
            No Google Earth Engine. No authentication. Just rasterio + your local Ollama instance.
        </div>
        <div style="margin-top:24px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap">
            <span class="badge badge-green">No GEE</span>
            <span class="badge badge-blue">COG windowed reads</span>
            <span class="badge badge-green">Local LLM</span>
            <span class="badge badge-blue">Change detection</span>
            <span class="badge badge-warn">PCA false-color</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Pipeline execution ────────────────────────────────────────────────────────
core = _import_core()
AOI = core["AOI"]

aoi = AOI(lon_min, lat_min, lon_max, lat_max, aoi_name)

status = st.empty()
progress = st.progress(0)

def step(msg, pct):
    status.markdown(f'<div class="mono" style="color:#00d4aa">⟳ {msg}</div>',
                    unsafe_allow_html=True)
    progress.progress(pct)

try:
    # 1. Index
    step("Loading tile index from Source Cooperative…", 5)
    index = _load_index()

    # 2. Find tiles Y1
    step(f"Discovering tiles for {aoi_name} / {year1}…", 15)
    tiles_y1 = core["find_tiles_for_aoi"](aoi, year1, index)

    if not tiles_y1:
        progress.empty()
        status.empty()
        st.error(f"No tiles found for this AOI in {year1}. Try a larger bounding box or different year.")
        st.stop()

    # 3. Read Y1
    step(f"Reading COG for {year1} (windowed VSICURL)…", 30)
    lon_c = (lon_min + lon_max) / 2
    lat_c = (lat_min + lat_max) / 2
    utm_zone = core["wgs84_to_utm_zone"](lon_c, lat_c)
    epsg     = core["utm_epsg"](utm_zone)

    url_y1 = core["build_tile_url"](tiles_y1[0]["filename"])
    emb_y1, mask_y1, info_y1 = core["read_tile_window"](url_y1, aoi, epsg)

    # 4. Optional Y2
    emb_y2 = mask_y2 = None
    if year2:
        step(f"Reading COG for {year2}…", 50)
        tiles_y2 = core["find_tiles_for_aoi"](aoi, year2, index)
        if tiles_y2:
            url_y2 = core["build_tile_url"](tiles_y2[0]["filename"])
            emb_y2, mask_y2, info_y2 = core["read_tile_window"](url_y2, aoi, epsg)

    # 5. Analysis
    step("Computing PCA false-color…", 65)
    pca_img_y1 = core["pca_rgb"](emb_y1, mask_y1) if mask_y1.sum() > 3 else None
    pca_img_y2 = core["pca_rgb"](emb_y2, mask_y2) if (emb_y2 is not None and mask_y2.sum() > 3) else None

    change_map = None
    if emb_y2 is not None:
        step("Computing cosine change map…", 75)
        change_map = core["cosine_change_map"](emb_y1, emb_y2)

    # 6. Summaries
    step("Summarizing embeddings…", 82)
    summary_y1 = core["summarize_embeddings"](emb_y1, mask_y1, aoi, year1, change_map)
    summary_y2 = None
    if emb_y2 is not None:
        summary_y2 = core["summarize_embeddings"](emb_y2, mask_y2, aoi, year2, change_map)

    # 7. Ollama
    llm_response = ""
    if not skip_llm:
        step(f"Querying {ollama_model} via Ollama…", 90)
        effective_task = "change" if (year2 and summary_y2) else task
        llm_response = core["query_ollama"](
            summary_y1,
            change_summary=summary_y2,
            model=ollama_model,
            host=ollama_host,
            api_key=ollama_api_key if ollama_api_key else None,
            task=effective_task,
        )

    progress.progress(100)
    status.empty()
    progress.empty()

except Exception as e:
    progress.empty()
    status.empty()
    st.error(f"Pipeline error: {e}")
    st.exception(e)
    st.stop()


# ── Results layout ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px">
  <span style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700">{aoi_name}</span>
  <span class="badge badge-green">{year1}</span>
  {"<span class='badge badge-warn'>→ " + str(year2) + "</span>" if year2 else ""}
  <span class="badge badge-blue">UTM {utm_zone} · EPSG:{epsg}</span>
</div>
""", unsafe_allow_html=True)

# ── Key metrics ───────────────────────────────────────────────────────────────
s = summary_y1
cov = s.get("coverage_pct", 0)
n_valid = s.get("pixel_count_valid", 0)
n_total = s.get("pixel_count_total", 1)
var = s.get("mean_intra_aoi_variance", 0)

cols_m = st.columns(4)
with cols_m[0]:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Valid Pixels</div>
      <div class="metric-value">{n_valid:,}</div>
      <div class="metric-sub">{cov}% coverage</div>
    </div>""", unsafe_allow_html=True)

with cols_m[1]:
    h, w = emb_y1.shape[:2]
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Window Shape</div>
      <div class="metric-value">{h}×{w}</div>
      <div class="metric-sub">pixels · 10m resolution</div>
    </div>""", unsafe_allow_html=True)

with cols_m[2]:
    top_ch = s.get("top3_active_channels", [0])
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Top Channel</div>
      <div class="metric-value">A{top_ch[0]:02d}</div>
      <div class="metric-sub">mean |{s.get("top3_channel_means", [0])[0]}|</div>
    </div>""", unsafe_allow_html=True)

with cols_m[3]:
    if change_map is not None and mask_y1 is not None:
        pct_hi = summary_y2.get("pct_high_change", 0) if summary_y2 else 0
        chg_mean = summary_y2.get("change_mean", 0) if summary_y2 else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">High Change</div>
          <div class="metric-value" style="color:var(--warn)">{pct_hi}%</div>
          <div class="metric-sub">mean cosine dist {chg_mean}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Intra-AOI Var</div>
          <div class="metric-value">{var:.4f}</div>
          <div class="metric-sub">mean channel std</div>
        </div>""", unsafe_allow_html=True)


# ── Visuals row ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Embedding Visualizations</div>', unsafe_allow_html=True)

if year2 and change_map is not None:
    vcol1, vcol2, vcol3 = st.columns(3)
else:
    vcol1, vcol2 = st.columns(2)

with vcol1:
    st.caption(f"PCA False-Color — {year1}")
    if pca_img_y1 is not None:
        st.image(pca_img_y1, use_container_width=True,
                 caption="PC1→R  PC2→G  PC3→B (64-dim AEF embeddings)")
    else:
        st.info("Insufficient valid pixels for PCA.")

with vcol2:
    st.caption(f"Channel Activation — {year1}")
    valid_emb = emb_y1[mask_y1]
    ch_means = np.abs(valid_emb.mean(axis=0))
    ch_max = ch_means.max() or 1.0

    # Build inline bar chart HTML for all 64 channels
    bars_html = '<div class="ch-grid">'
    # Show top 20 by activation to keep it readable
    top_idx = np.argsort(ch_means)[::-1][:20]
    for i in top_idx:
        pct = int(ch_means[i] / ch_max * 100)
        bars_html += f"""
        <div class="ch-row">
          <div class="ch-label">A{i:02d}</div>
          <div class="ch-bar-wrap"><div class="ch-bar" style="width:{pct}%"></div></div>
          <div class="ch-val">{ch_means[i]:.4f}</div>
        </div>"""
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)
    st.caption("Top 20 channels by |mean activation|")

if year2 and change_map is not None:
    with vcol3:
        st.caption(f"Cosine Change Map — {year1} → {year2}")
        joint_mask = mask_y1 & mask_y2 if mask_y2 is not None else mask_y1
        chg = np.where(joint_mask, change_map, np.nan)

        # Normalize to 0–255 for display, colormap: green→yellow→red
        chg_min = np.nanpercentile(chg, 2)
        chg_max = np.nanpercentile(chg, 98)
        chg_norm = np.clip((chg - chg_min) / max(chg_max - chg_min, 1e-8), 0, 1)

        # Apply RdYlGn_r colormap manually
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            cmap = cm.get_cmap("RdYlGn_r")
            chg_rgba = (cmap(np.nan_to_num(chg_norm, nan=0.5)) * 255).astype(np.uint8)
            # mask invalid pixels to black
            chg_rgba[np.isnan(chg)] = [20, 25, 40, 255]
            st.image(chg_rgba[:, :, :3], use_container_width=True,
                     caption=f"1 − cos(t1,t2) · green=stable · red=changed")
        except Exception:
            st.info("Install matplotlib for change map visualization.")

        if year2 and summary_y2:
            cols_ch = st.columns(3)
            cols_ch[0].metric("Mean Δ", f"{summary_y2.get('change_mean',0):.4f}")
            cols_ch[1].metric("P90 Δ",  f"{summary_y2.get('change_p90',0):.4f}")
            cols_ch[2].metric("Hi-chg", f"{summary_y2.get('pct_high_change',0)}%")

        if year2 and pca_img_y2 is not None:
            with st.expander(f"PCA False-Color {year2}"):
                st.image(pca_img_y2, use_container_width=True)


# ── LLM response ──────────────────────────────────────────────────────────────
if not skip_llm and llm_response:
    st.markdown('<div class="section-head">Ollama Interpretation</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px">
      <span style="font-size:1.1rem">🤖</span>
      <span class="mono" style="color:#64748b">{ollama_model} · {task_label.lower()}</span>
      <span class="badge {'badge-green' if ollama_api_key else 'badge-blue'}">{'cloud' if ollama_api_key else 'local'}</span>
    </div>
    <div class="llm-box">{llm_response}</div>
    """, unsafe_allow_html=True)
elif skip_llm:
    st.info("Ollama skipped — embedding extraction complete.")
elif llm_response == "":
    st.warning("Ollama returned an empty response. Check that `ollama serve` is running and the model is pulled.")


# ── Raw summary JSON ──────────────────────────────────────────────────────────
with st.expander("📋 Raw Embedding Summary (JSON)"):
    out = {"year1": summary_y1}
    if summary_y2:
        out["year2"] = summary_y2
    st.json(out)
    st.download_button(
        "Download JSON",
        data=json.dumps(out, indent=2),
        file_name=f"aef_{aoi_name.replace(' ','_')}_{year1}.json",
        mime="application/json",
    )

if pca_img_y1 is not None:
    try:
        from PIL import Image as PILImage
        import io
        buf = io.BytesIO()
        PILImage.fromarray(pca_img_y1).save(buf, format="PNG")
        st.download_button(
            "⬇ Download PCA PNG",
            data=buf.getvalue(),
            file_name=f"aef_pca_{aoi_name.replace(' ','_')}_{year1}.png",
            mime="image/png",
        )
    except ImportError:
        pass


# ── Attribution footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.68rem; color:#475569; line-height:1.6; text-align:center">
  <em>"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."</em><br>
  Data hosted at <code>source.coop/tge-labs/aef</code> · CC-BY 4.0 ·
  Built without Google Earth Engine
</div>
""", unsafe_allow_html=True)
