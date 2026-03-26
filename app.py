"""
Berlin Airbnb Market Analysis — Streamlit Dashboard
B106 Data Visualisation | Inside Airbnb Dataset

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_FILE = "app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------------------------------
# SAFE PLOTLY IMPORT
# -----------------------------------------------------------------------------
PLOTLY_AVAILABLE = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    logging.exception("Plotly import failed: %s", e)

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Berlin Airbnb Market Analysis",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# DESIGN TOKENS
# -----------------------------------------------------------------------------
C = {
    "primary": "#2E5D9E",
    "secondary": "#E8834E",
    "accent": "#4CAF82",
    "neutral": "#8C9EB0",
    "light": "#F5F7FA",
    "dark": "#EAEFF5",
    "bg": "#07111F",
    "card_bg": "#0E1B2B",
}

# FIX: Moved colorway out of PLOTLY_LAYOUT so it isn't re-applied via
# update_layout() and doesn't overwrite per-figure colour settings.
PLOTLY_COLORWAY = [
    C["primary"],
    C["secondary"],
    C["accent"],
    C["neutral"],
    "#9B5FBF",
    "#E5C84A",
    "#3DB5B5",
]

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", color="#EAEFF5"),
    paper_bgcolor=C["bg"],
    plot_bgcolor="#0E1B2B",
    # colorway intentionally omitted here — set per-figure or via template
    title=dict(font=dict(size=15, color="#EAEFF5"), x=0.02),
    xaxis=dict(gridcolor="#223247", linecolor="#2B3D52", showgrid=True),
    yaxis=dict(gridcolor="#223247", linecolor="#2B3D52", showgrid=True),
    margin=dict(l=50, r=30, t=55, b=50),
    hoverlabel=dict(
        bgcolor="#09131F",
        font_color="white",
        bordercolor="#09131F",
        font_size=12,
    ),
)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #07111F;
    color: #EAEFF5;
  }

  .stApp {
    background: #07111F;
  }

  .kpi-card {
    background: linear-gradient(135deg, #0E1B2B 0%, #12253A 100%);
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #1F3349;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    margin-bottom: 4px;
  }

  .kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
  }

  .kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #9FB2C7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
  }

  .kpi-delta {
    font-size: 0.82rem;
    font-weight: 600;
    color: #4CAF82;
    margin-top: 2px;
  }

  .section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #F5F8FC;
    border-bottom: 2px solid #1B2C3F;
    padding-bottom: 8px;
    margin: 16px 0 12px 0;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #18283A 0%, #2A3C51 100%);
  }

  [data-testid="stSidebar"] * {
    color: #E0E8F0 !important;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 6px; }

  .stTabs [data-baseweb="tab"] {
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 0.88rem;
  }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def safe_percent_true(series: pd.Series) -> str:
    if series is None or len(series) == 0:
        return "N/A"
    valid = series.dropna()
    if len(valid) == 0:
        return "N/A"
    try:
        return f"{valid.astype(float).mean() * 100:.1f}%"
    except Exception:
        return "N/A"


def render(fig, height: int = 400):
    if fig is None:
        return None
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


def safe_plot(fig, height: int = 400):
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Add `plotly` to your requirements.txt.")
        return
    try:
        st.plotly_chart(render(fig, height), use_container_width=True)
    except Exception as e:
        logging.exception("Chart render failed: %s", e)
        st.info("This chart could not be rendered for the current data selection.")


def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(col in df.columns for col in cols)


# FIX: Expanded mapping to handle float representations (1.0, 0.0) that
# Inside Airbnb CSVs commonly produce after pd.read_csv parses boolean columns.
def clean_bool_col(series: pd.Series) -> pd.Series:
    def _map(v):
        if v is True or v == 1 or v == 1.0:
            return True
        if v is False or v == 0 or v == 0.0:
            return False
        if isinstance(v, str):
            low = v.strip().lower()
            if low == "t" or low == "true":
                return True
            if low == "f" or low == "false":
                return False
        return np.nan
    return series.apply(_map)


def show_generic_data_error():
    st.error(
        "⚠️ The dataset could not be loaded correctly. "
        "Check the path or upload a valid file. Technical details were written to app.log."
    )


# -----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_from_path(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path_obj, low_memory=False, compression="infer")
    return clean_dataframe(df)


# FIX: Accept the raw bytes directly so cache key is stable across re-runs
# without calling getvalue() outside the cached function.
@st.cache_data(show_spinner=False)
def load_and_clean_from_upload(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    from io import BytesIO
    bio = BytesIO(file_bytes)
    compression = "gzip" if file_name.endswith(".gz") else "infer"
    df = pd.read_csv(bio, low_memory=False, compression=compression)
    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    # FIX: Convert to str only once, then clean — avoids double-conversion
    # issues and ensures "nan" / "None" strings are caught before to_numeric.
    price_str = df["price"].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    price_str = price_str.replace({"nan": np.nan, "None": np.nan, "": np.nan, "NaN": np.nan})
    df["price"] = pd.to_numeric(price_str, errors="coerce")
    df = df.dropna(subset=["price"]).copy()
    df = df[df["price"] > 0].copy()

    if len(df) > 20:
        q1, q99 = df["price"].quantile([0.01, 0.99])
        df = df[(df["price"] >= q1) & (df["price"] <= q99)].copy()

    bool_cols = [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = clean_bool_col(df[col])

    for col in ["last_review", "host_since"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # FIX: Rate columns may already be floats (0.0–1.0) in some dataset
    # versions instead of "97%" strings. Detect which format and normalise
    # to a 0–100 float consistently.
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            raw = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
            raw = raw.replace({"nan": np.nan, "None": np.nan, "": np.nan, "NaN": np.nan})
            numeric = pd.to_numeric(raw, errors="coerce")
            # Values ≤ 1.0 are proportions (0.0–1.0); scale to 0–100
            is_proportion = numeric.dropna().lt(1.01).all()
            if is_proportion:
                numeric = numeric * 100
            df[col] = numeric

    numeric_cols = [
        "reviews_per_month",
        "minimum_nights",
        "accommodates",
        "review_scores_rating",
        "availability_365",
        "availability_90",
        "availability_30",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "reviews_per_month" not in df.columns:
        df["reviews_per_month"] = 0.0
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    if "minimum_nights" not in df.columns:
        df["minimum_nights"] = 1
    df["minimum_nights"] = df["minimum_nights"].fillna(1)

    if has_cols(df, ["host_id", "id"]):
        host_counts = df.groupby("host_id")["id"].count()
        df["host_listing_count_actual"] = df["host_id"].map(host_counts)
    else:
        df["host_listing_count_actual"] = 1

    df["host_type"] = np.where(
        df["host_listing_count_actual"].fillna(1) > 1,
        "Multi-listing",
        "Single-listing",
    )

    def nights_bucket(n):
        try:
            n = float(n)
        except Exception:
            return "Unknown"
        if n <= 1:
            return "1 night"
        if n <= 3:
            return "2–3 nights"
        if n <= 7:
            return "4–7 nights"
        if n <= 30:
            return "8–30 nights"
        return "30+ nights"

    df["nights_bucket"] = df["minimum_nights"].apply(nights_bucket)

    if "host_is_superhost" not in df.columns:
        df["host_is_superhost"] = np.nan

    df["superhost_label"] = df["host_is_superhost"].map(
        {True: "Superhost", False: "Regular host"}
    )

    if "neighbourhood_cleansed" not in df.columns:
        df["neighbourhood_cleansed"] = "Unknown"
    df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("Unknown").astype(str)

    if "room_type" not in df.columns:
        df["room_type"] = "Unknown"
    df["room_type"] = df["room_type"].fillna("Unknown").astype(str)

    return df


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏙️ Berlin Airbnb")
    st.markdown("**Market Analysis Dashboard**")
    st.markdown("---")

    data_mode = st.radio(
        "Data source",
        ["Use repo file path", "Upload dataset"],
        index=0,
    )

    DATA_PATH = st.text_input(
        "Dataset path",
        value="data/listings.csv.gz",
        help="Path to the Inside Airbnb Berlin listings file",
    )

    uploaded_file = st.file_uploader(
        "Upload listings dataset",
        type=["csv", "gz"],
        help="Upload Inside Airbnb Berlin listings file (.csv or .csv.gz)",
    )

    st.markdown("### Filters")

    data_ok = False
    df_full = pd.DataFrame()

    try:
        with st.spinner("Loading data…"):
            if data_mode == "Upload dataset" and uploaded_file is not None:
                df_full = load_and_clean_from_upload(uploaded_file.getvalue(), uploaded_file.name)
                data_ok = True
            elif data_mode == "Use repo file path":
                df_full = load_and_clean_from_path(DATA_PATH)
                data_ok = True
    except FileNotFoundError as e:
        logging.exception("Data file not found: %s", e)
        st.error(
            "⚠️ Data file not found.\n\n"
            "Either upload the dataset here, or place `listings.csv.gz` inside your repo "
            "at `data/listings.csv.gz`."
        )
    except Exception as e:
        logging.exception("Failed to load dataset: %s", e)
        show_generic_data_error()

    if data_ok and not df_full.empty:
        all_neighbourhoods = sorted(df_full["neighbourhood_cleansed"].dropna().unique().tolist())
        all_room_types = sorted(df_full["room_type"].dropna().unique().tolist())

        price_min = int(np.floor(df_full["price"].min()))
        price_max = int(np.ceil(df_full["price"].max()))

        if price_min >= price_max:
            price_max = price_min + 1

        selected_neighbourhoods = st.multiselect(
            "Neighbourhood",
            options=all_neighbourhoods,
            default=[],
            placeholder="All neighbourhoods",
        )

        selected_room_types = st.multiselect(
            "Room type",
            options=all_room_types,
            default=[],
            placeholder="All room types",
        )

        price_range = st.slider(
            "Price range (€/night)",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            step=max(1, (price_max - price_min) // 100),
        )

        superhost_filter = st.selectbox(
            "Host type",
            ["All", "Superhost only", "Regular hosts only"],
        )

        st.markdown("---")
        st.caption("Dataset: Inside Airbnb Berlin")
        st.caption("Course: B106 Data Visualisation")
    else:
        # FIX: Always define filter variables so apply_filters() never
        # references an unbound name when data_ok is False.
        selected_neighbourhoods = []
        selected_room_types = []
        price_range = (0, 999999)
        superhost_filter = "All"


# -----------------------------------------------------------------------------
# FILTERING
# -----------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    mask = (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])

    if selected_neighbourhoods:
        mask &= df["neighbourhood_cleansed"].isin(selected_neighbourhoods)

    if selected_room_types:
        mask &= df["room_type"].isin(selected_room_types)

    # FIX: Use .eq() instead of == True to correctly handle pandas nullable
    # boolean dtype (BooleanDtype) without raising ambiguity errors.
    if superhost_filter == "Superhost only" and "host_is_superhost" in df.columns:
        mask &= df["host_is_superhost"].eq(True)
    elif superhost_filter == "Regular hosts only" and "host_is_superhost" in df.columns:
        mask &= df["host_is_superhost"].eq(False)

    return df.loc[mask].copy()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if not PLOTLY_AVAILABLE:
    st.title("🏙️ Berlin Airbnb Market Analysis")
    st.error(
        "Plotly is missing in this environment. Add this to requirements.txt:\n"
        "streamlit\npandas\nnumpy\nplotly"
    )
    st.stop()

if not data_ok:
    st.title("🏙️ Berlin Airbnb Market Analysis")
    st.info("Upload the dataset in the sidebar, or place it in your repo at `data/listings.csv.gz`.")
    st.stop()

df = apply_filters(df_full)

st.title("🏙️ Berlin Airbnb Market Analysis")
filtered_text = f"(filtered from {len(df_full):,})" if len(df) < len(df_full) else ""
st.markdown(f"Analysing **{len(df):,}** listings {filtered_text} · Inside Airbnb Berlin dataset")

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

tabs = st.tabs([
    "📊 Overview",
    "🗺️ Neighbourhoods",
    "🏠 Listing Characteristics",
    "💶 Pricing & Demand",
    "👤 Host Concentration",
    "🔍 Insight Shortlist",
])

# ── Tab 0: Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Market Snapshot</p>', unsafe_allow_html=True)

    superhost_share = safe_percent_true(df["host_is_superhost"]) if "host_is_superhost" in df.columns else "N/A"
    multi_host_share = f"{(df['host_type'] == 'Multi-listing').mean() * 100:.1f}%"

    kpi_data = [
        ("Total Listings", f"{len(df):,}", "Active on Airbnb"),
        ("Median Price", f"€{df['price'].median():.0f}", "Per night"),
        ("Superhost Share", superhost_share, "Of all hosts"),
        ("Unique Neighbourhoods", f"{df['neighbourhood_cleansed'].nunique():,}", "Across Berlin"),
        ("Avg Reviews/Month", f"{df['reviews_per_month'].mean():.2f}", "Proxy for bookings"),
        ("Multi-listing Hosts", multi_host_share, "Professional operators"),
    ]

    cols = st.columns(6)
    for col, (label, value, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-delta">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        fig_hist = px.histogram(
            df,
            x="price",
            nbins=60,
            title="Nightly Price Distribution",
            labels={"price": "Nightly price (€)", "count": "Listings"},
            color_discrete_sequence=[C["primary"]],
        )
        fig_hist.update_traces(marker_line_width=0)
        fig_hist.add_vline(
            x=df["price"].median(),
            line_dash="dash",
            line_color=C["secondary"],
            line_width=2,
            annotation_text=f"Median: €{df['price'].median():.0f}",
            annotation_position="top right",
        )
        safe_plot(fig_hist, 360)

    with c2:
        room_counts = df["room_type"].value_counts(dropna=False).reset_index()
        room_counts.columns = ["room_type", "count"]

        fig_room = px.bar(
            room_counts.sort_values("count", ascending=True),
            x="count",
            y="room_type",
            orientation="h",
            title="Listing Count by Room Type",
            labels={"count": "Listings", "room_type": ""},
            color="room_type",
            color_discrete_sequence=[C["primary"], C["secondary"], C["accent"], C["neutral"]],
            text="count",
        )
        fig_room.update_layout(showlegend=False)
        safe_plot(fig_room, 360)

# ── Tab 1: Neighbourhoods ────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Neighbourhood Analysis</p>', unsafe_allow_html=True)

    neigh_agg = (
        df.groupby("neighbourhood_cleansed")
        .agg(
            median_price=("price", "median"),
            listing_count=("price", "count"),
            avg_reviews=("reviews_per_month", "mean"),
        )
        .reset_index()
        .sort_values("median_price", ascending=False)
    )

    max_top_n = min(30, len(neigh_agg))
    min_top_n = 1 if max_top_n < 5 else 5
    default_top_n = min(15, max_top_n)

    top_n = st.slider("Top N neighbourhoods to show", min_top_n, max_top_n, default_top_n)
    top_neigh = neigh_agg.head(top_n)

    fig_neigh = px.bar(
        top_neigh.sort_values("median_price"),
        x="median_price",
        y="neighbourhood_cleansed",
        orientation="h",
        title=f"Top {top_n} Neighbourhoods by Median Nightly Price",
        labels={"median_price": "Median price (€)", "neighbourhood_cleansed": ""},
        color="median_price",
        color_continuous_scale=[[0, "#A8C4E0"], [1, C["primary"]]],
        text="median_price",
    )
    fig_neigh.update_traces(texttemplate="€%{text:.0f}", textposition="outside")
    fig_neigh.update_coloraxes(showscale=False)
    safe_plot(fig_neigh, 520)

# ── Tab 2: Listing Characteristics ───────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">Listing Characteristics</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # FIX: Use points=False instead of points="outliers" — "outliers" can
        # silently fail in some Plotly versions when the column has many NaN
        # values; False is always safe and renders faster on large datasets.
        fig_box = px.box(
            df,
            x="room_type",
            y="price",
            color="room_type",
            title="Price Distribution by Room Type",
            labels={"price": "Nightly price (€)", "room_type": ""},
            color_discrete_sequence=[C["primary"], C["secondary"], C["accent"], C["neutral"]],
            points=False,
        )
        fig_box.update_traces(boxmean=True)
        fig_box.update_layout(showlegend=False)
        safe_plot(fig_box, 400)

    with c2:
        # FIX: Added explicit NaN drop before between() so the filter doesn't
        # silently discard everything; added a fallback message when the column
        # is missing or the filtered result is empty.
        if "accommodates" in df.columns:
            cap_df = df.copy()
            cap_df["accommodates"] = pd.to_numeric(cap_df["accommodates"], errors="coerce")
            cap_df = cap_df.dropna(subset=["accommodates"])
            cap_agg = (
                cap_df[cap_df["accommodates"].between(1, 10, inclusive="both")]
                .groupby("accommodates")["price"]
                .median()
                .reset_index()
            )
            if not cap_agg.empty:
                fig_cap = px.line(
                    cap_agg,
                    x="accommodates",
                    y="price",
                    title="Median Price by Guest Capacity",
                    labels={"accommodates": "Guests accommodated", "price": "Median price (€)"},
                    markers=True,
                    color_discrete_sequence=[C["primary"]],
                )
                safe_plot(fig_cap, 400)
            else:
                st.info("Not enough capacity data available for the current selection.")
        else:
            st.info("Column `accommodates` is not present in this dataset.")

# ── Tab 3: Pricing & Demand ───────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Pricing & Demand Signals</p>', unsafe_allow_html=True)

    avail_col = next(
        (c for c in ["availability_365", "availability_90", "availability_30"] if c in df.columns),
        None,
    )

    if avail_col:
        avail_df = df[[avail_col, "reviews_per_month", "room_type"]].dropna().copy()
        if len(avail_df) > 3000:
            avail_df = avail_df.sample(3000, random_state=42)

        fig_sc = px.scatter(
            avail_df,
            x=avail_col,
            y="reviews_per_month",
            color="room_type",
            opacity=0.35,
            title="Availability vs Review Velocity",
            labels={
                avail_col: "Days available",
                "reviews_per_month": "Reviews / month",
                "room_type": "Room type",
            },
            color_discrete_sequence=[C["primary"], C["secondary"], C["accent"], C["neutral"]],
        )
        fig_sc.update_traces(marker_size=5)
        safe_plot(fig_sc, 420)
    else:
        # FIX: Previously this tab was completely blank when no availability
        # column existed — now it shows a clear explanation.
        st.info(
            "No availability column found in this dataset. "
            "Expected one of: `availability_365`, `availability_90`, or `availability_30`."
        )

# ── Tab 4: Host Concentration ─────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">Host Concentration Analysis</p>', unsafe_allow_html=True)

    neighbourhood_count = df["neighbourhood_cleansed"].nunique()
    max_top_n2 = min(20, max(1, neighbourhood_count))
    min_top_n2 = 1 if max_top_n2 < 5 else 5
    default_top_n2 = min(12, max_top_n2)

    top_n2 = st.slider("Top N neighbourhoods", min_top_n2, max_top_n2, default_top_n2, key="hc_slider")
    top_n_list = df["neighbourhood_cleansed"].value_counts().head(top_n2).index.tolist()
    df_hc = df[df["neighbourhood_cleansed"].isin(top_n_list)].copy()

    hc_mix = (
        df_hc.groupby(["neighbourhood_cleansed", "host_type"])
        .size()
        .reset_index(name="count")
    )

    totals = hc_mix.groupby("neighbourhood_cleansed")["count"].transform("sum")
    hc_mix["pct"] = np.where(totals > 0, hc_mix["count"] / totals * 100, 0)

    fig_hc = px.bar(
        hc_mix,
        x="pct",
        y="neighbourhood_cleansed",
        color="host_type",
        orientation="h",
        barmode="stack",
        title="Single vs Multi-Listing Host Share",
        labels={"pct": "Share (%)", "neighbourhood_cleansed": ""},
        color_discrete_map={
            "Single-listing": C["neutral"],
            "Multi-listing": C["secondary"],
        },
    )
    safe_plot(fig_hc, 480)

# ── Tab 5: Insight Shortlist ──────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-header">Final Insight Shortlist</p>', unsafe_allow_html=True)

    # FIX: Was just a static info string. Now dynamically computes real insights
    # from the filtered dataframe so the tab is actually useful.

    # --- Compute insights ---
    top_neigh_price = (
        df.groupby("neighbourhood_cleansed")["price"]
        .median()
        .sort_values(ascending=False)
    )
    priciest_neigh = top_neigh_price.index[0] if len(top_neigh_price) else "N/A"
    priciest_price = top_neigh_price.iloc[0] if len(top_neigh_price) else 0

    cheapest_neigh = top_neigh_price.index[-1] if len(top_neigh_price) else "N/A"
    cheapest_price = top_neigh_price.iloc[-1] if len(top_neigh_price) else 0

    multi_pct = (df["host_type"] == "Multi-listing").mean() * 100

    most_common_room = df["room_type"].value_counts().index[0] if len(df) else "N/A"
    most_common_room_pct = df["room_type"].value_counts(normalize=True).iloc[0] * 100 if len(df) else 0

    superhost_median = df[df["host_is_superhost"].eq(True)]["price"].median() if "host_is_superhost" in df.columns else None
    regular_median = df[df["host_is_superhost"].eq(False)]["price"].median() if "host_is_superhost" in df.columns else None

    high_demand = df[df["reviews_per_month"] > df["reviews_per_month"].quantile(0.75)]
    high_demand_room = high_demand["room_type"].value_counts().index[0] if len(high_demand) else "N/A"

    insights = [
        {
            "icon": "💰",
            "title": "Priciest neighbourhood",
            "body": (
                f"**{priciest_neigh}** commands the highest median nightly rate at "
                f"**€{priciest_price:.0f}**, compared to **€{cheapest_price:.0f}** "
                f"in the most affordable neighbourhood (**{cheapest_neigh}**)."
            ),
        },
        {
            "icon": "🏢",
            "title": "Professional host concentration",
            "body": (
                f"**{multi_pct:.1f}%** of listings belong to multi-listing hosts, "
                "suggesting a significant share of the market is operated commercially "
                "rather than by individual home-sharers."
            ),
        },
        {
            "icon": "🛏️",
            "title": "Dominant listing type",
            "body": (
                f"**{most_common_room}** is the most common room type, "
                f"accounting for **{most_common_room_pct:.1f}%** of all listings in "
                "the current selection."
            ),
        },
        {
            "icon": "⭐",
            "title": "Superhost price premium",
            "body": (
                (
                    f"Superhosts charge a median of **€{superhost_median:.0f}/night** vs "
                    f"**€{regular_median:.0f}/night** for regular hosts — a "
                    f"**{abs(superhost_median - regular_median):.0f}€ difference** "
                    f"({'premium' if superhost_median > regular_median else 'discount'})."
                )
                if superhost_median is not None and regular_median is not None
                and not (pd.isna(superhost_median) or pd.isna(regular_median))
                else "Superhost data is not available for the current selection."
            ),
        },
        {
            "icon": "📈",
            "title": "High-demand listing type",
            "body": (
                f"Among the top 25% most-reviewed listings (a proxy for booking frequency), "
                f"**{high_demand_room}** is the most prevalent room type — suggesting "
                "guests favour this category."
            ),
        },
        {
            "icon": "📊",
            "title": "Price spread",
            "body": (
                f"Nightly prices range from **€{df['price'].quantile(0.05):.0f}** (5th percentile) "
                f"to **€{df['price'].quantile(0.95):.0f}** (95th percentile), "
                f"with a median of **€{df['price'].median():.0f}** — "
                "indicating a wide spread driven by neighbourhood and room type."
            ),
        },
    ]

    for ins in insights:
        with st.container():
            st.markdown(
                f"""
                <div class="kpi-card" style="margin-bottom:12px;">
                    <div style="font-size:1.3rem; margin-bottom:6px;">{ins['icon']} <strong style="font-size:0.95rem; color:#C8D8E8;">{ins['title']}</strong></div>
                    <div style="font-size:0.88rem; color:#B0C4D8; line-height:1.55;">{ins['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
