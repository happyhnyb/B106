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
    "dark": "#1C2B3A",
    "bg": "#FFFFFF",
    "card_bg": "#F0F4FA",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", color=C["dark"]),
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["light"],
    colorway=[
        C["primary"],
        C["secondary"],
        C["accent"],
        C["neutral"],
        "#9B5FBF",
        "#E5C84A",
        "#3DB5B5",
    ],
    title=dict(font=dict(size=15, color=C["dark"]), x=0.02),
    xaxis=dict(gridcolor="#E8EEF4", linecolor="#D0D8E0", showgrid=True),
    yaxis=dict(gridcolor="#E8EEF4", linecolor="#D0D8E0", showgrid=True),
    margin=dict(l=50, r=30, t=55, b=50),
    hoverlabel=dict(
        bgcolor=C["dark"],
        font_color="white",
        bordercolor=C["dark"],
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
  }

  .kpi-card {
    background: linear-gradient(135deg, #F0F4FA 0%, #E8EEF8 100%);
    border-radius: 12px;
    padding: 18px 20px;
    border-left: 4px solid #2E5D9E;
    box-shadow: 0 2px 8px rgba(46,93,158,0.08);
    margin-bottom: 4px;
  }

  .kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #2E5D9E;
    line-height: 1.1;
  }

  .kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #8C9EB0;
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
    color: #1C2B3A;
    border-bottom: 2px solid #E8EEF4;
    padding-bottom: 8px;
    margin: 16px 0 12px 0;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1C2B3A 0%, #2E3D50 100%);
  }

  [data-testid="stSidebar"] * {
    color: #E0E8F0 !important;
  }

  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stTextInput label {
    color: #A8BDD0 !important;
    font-size: 0.82rem;
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


def clean_bool_col(series: pd.Series) -> pd.Series:
    mapping = {
        "t": True, "f": False,
        "true": True, "false": False,
        "True": True, "False": False,
        True: True, False: False,
        1: True, 0: False,
    }
    return series.map(mapping)


def show_generic_data_error():
    st.error(
        "⚠️ The dataset could not be loaded correctly. "
        "Check the path and file format. Technical details were written to app.log."
    )


# -----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path_obj, low_memory=False, compression="infer")

    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    # Price cleaning
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()

    # Remove impossible or very low prices
    df = df[df["price"] > 0].copy()

    # Trim extreme outliers
    if len(df) > 20:
        q1, q99 = df["price"].quantile([0.01, 0.99])
        df = df[(df["price"] >= q1) & (df["price"] <= q99)].copy()

    # Boolean columns
    bool_cols = [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = clean_bool_col(df[col])

    # Date columns
    for col in ["last_review", "host_since"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Rate columns
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Numeric columns that are often useful
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

    # Host counts
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

    # Nights bucket
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

    # Superhost label
    if "host_is_superhost" not in df.columns:
        df["host_is_superhost"] = np.nan

    df["superhost_label"] = df["host_is_superhost"].map(
        {True: "Superhost", False: "Regular host"}
    )

    # Required text cols
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

    DATA_PATH = st.text_input(
        "Dataset path",
        value="data/listings.csv.gz",
        help="Path to the Inside Airbnb Berlin listings file",
    )

    st.markdown("### Filters")

    try:
        with st.spinner("Loading data…"):
            df_full = load_and_clean(DATA_PATH)
        data_ok = True
    except FileNotFoundError as e:
        logging.exception("Data file not found: %s", e)
        st.error(
            "⚠️ Data file not found.\n\n"
            "Download `listings.csv.gz` from Inside Airbnb and place it in your repo, "
            "for example in `data/listings.csv.gz`."
        )
        data_ok = False
        df_full = pd.DataFrame()
    except Exception as e:
        logging.exception("Failed to load dataset: %s", e)
        show_generic_data_error()
        data_ok = False
        df_full = pd.DataFrame()

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

    if superhost_filter == "Superhost only" and "host_is_superhost" in df.columns:
        mask &= df["host_is_superhost"] == True
    elif superhost_filter == "Regular hosts only" and "host_is_superhost" in df.columns:
        mask &= df["host_is_superhost"] == False

    return df.loc[mask].copy()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if not PLOTLY_AVAILABLE:
    st.title("🏙️ Berlin Airbnb Market Analysis")
    st.error(
        "Plotly is missing in this environment. Add it to `requirements.txt`:\n\n"
        "`streamlit`\n`pandas`\n`numpy`\n`plotly`"
    )
    st.stop()

if not data_ok:
    st.title("🏙️ Berlin Airbnb Market Analysis")
    st.info("Add a valid dataset path in the sidebar to begin.")
    st.stop()

df = apply_filters(df_full)

st.title("🏙️ Berlin Short-Term Rental Market")

filtered_text = f"(filtered from {len(df_full):,})" if len(df) < len(df_full) else ""
st.markdown(
    f"Analysing **{len(df):,}** listings {filtered_text} · Inside Airbnb Berlin dataset"
)

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "🗺️ Neighbourhoods",
    "🏠 Listing Characteristics",
    "💶 Pricing & Demand",
    "👤 Host Concentration",
    "🔍 Insight Shortlist",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
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

        if len(room_counts) > 0:
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
        else:
            st.info("No room type data available.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — NEIGHBOURHOODS
# ═════════════════════════════════════════════════════════════════════════════
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

    if neigh_agg.empty:
        st.info("No neighbourhood data available.")
    else:
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

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            bubble_df = neigh_agg.head(min(20, len(neigh_agg))).copy()
            fig_bubble = px.scatter(
                bubble_df,
                x="median_price",
                y="avg_reviews",
                size="listing_count",
                text="neighbourhood_cleansed",
                title="Price vs Review Velocity — Top Neighbourhoods",
                labels={
                    "median_price": "Median price (€)",
                    "avg_reviews": "Avg reviews/month",
                    "listing_count": "Listing count",
                },
                color="listing_count",
                color_continuous_scale=[[0, "#A8C4E0"], [1, C["primary"]]],
            )
            fig_bubble.update_traces(textposition="top center", textfont_size=9)
            fig_bubble.update_coloraxes(showscale=False)
            safe_plot(fig_bubble, 420)

        with c2:
            top_12 = neigh_agg.head(min(12, len(neigh_agg)))["neighbourhood_cleansed"].tolist()
            rt_neigh = (
                df[df["neighbourhood_cleansed"].isin(top_12)]
                .groupby(["neighbourhood_cleansed", "room_type"])
                .size()
                .reset_index(name="count")
            )

            if not rt_neigh.empty:
                total_per_neigh = rt_neigh.groupby("neighbourhood_cleansed")["count"].transform("sum")
                rt_neigh["pct"] = np.where(total_per_neigh > 0, rt_neigh["count"] / total_per_neigh * 100, 0)

                fig_rt = px.bar(
                    rt_neigh,
                    x="pct",
                    y="neighbourhood_cleansed",
                    color="room_type",
                    orientation="h",
                    barmode="stack",
                    title="Room Type Mix — Top Neighbourhoods",
                    labels={"pct": "Share (%)", "neighbourhood_cleansed": ""},
                    color_discrete_sequence=[C["primary"], C["secondary"], C["accent"], C["neutral"]],
                )
                safe_plot(fig_rt, 420)
            else:
                st.info("No room type mix available for current filters.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — LISTING CHARACTERISTICS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header">Listing Characteristics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_box = px.box(
            df,
            x="room_type",
            y="price",
            color="room_type",
            title="Price Distribution by Room Type",
            labels={"price": "Nightly price (€)", "room_type": ""},
            color_discrete_sequence=[C["primary"], C["secondary"], C["accent"], C["neutral"]],
            points="outliers",
        )
        fig_box.update_traces(boxmean=True)
        fig_box.update_layout(showlegend=False)
        safe_plot(fig_box, 400)

    with c2:
        if "accommodates" in df.columns:
            cap_df = df.copy()
            cap_df["accommodates"] = pd.to_numeric(cap_df["accommodates"], errors="coerce")
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
                st.info("No valid accommodates data available.")
        else:
            st.info("Column `accommodates` not available in this dataset.")

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        mn_order = ["1 night", "2–3 nights", "4–7 nights", "8–30 nights", "30+ nights"]
        mn_agg = (
            df.groupby("nights_bucket")
            .agg(median_price=("price", "median"), count=("price", "count"))
            .reindex(mn_order)
            .dropna()
            .reset_index()
        )

        if not mn_agg.empty:
            fig_mn = px.bar(
                mn_agg,
                x="nights_bucket",
                y="median_price",
                title="Median Price by Minimum-Night Policy",
                labels={"nights_bucket": "Minimum nights", "median_price": "Median price (€)"},
                color="median_price",
                color_continuous_scale=[[0, "#A8C4E0"], [1, C["primary"]]],
                text="median_price",
            )
            fig_mn.update_traces(texttemplate="€%{text:.0f}", textposition="outside")
            fig_mn.update_coloraxes(showscale=False)
            safe_plot(fig_mn, 380)
        else:
            st.info("No minimum-night data available.")

    with c4:
        if "review_scores_rating" in df.columns:
            rat_df = df.dropna(subset=["review_scores_rating"]).copy()
            if not rat_df.empty:
                fig_rat = px.histogram(
                    rat_df,
                    x="review_scores_rating",
                    nbins=30,
                    title="Distribution of Review Scores",
                    labels={"review_scores_rating": "Review score", "count": "Listings"},
                    color_discrete_sequence=[C["accent"]],
                )
                safe_plot(fig_rat, 380)
            else:
                st.info("No review score data available.")
        else:
            st.info("Column `review_scores_rating` not available.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRICING & DEMAND
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header">Pricing & Demand Signals</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        avail_col = next(
            (c for c in ["availability_365", "availability_90", "availability_30"] if c in df.columns),
            None,
        )

        if avail_col:
            avail_df = df[[avail_col, "reviews_per_month", "room_type"]].dropna().copy()
            avail_df = avail_df[avail_df[avail_col] >= 0]

            if len(avail_df) > 3000:
                avail_df = avail_df.sample(3000, random_state=42)

            if not avail_df.empty:
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
                st.info("No availability data available.")
        else:
            st.info("No availability column found in the dataset.")

    with c2:
        sh_df = df.dropna(subset=["superhost_label"]).copy()

        if not sh_df.empty:
            pctile_df = (
                sh_df.groupby("superhost_label")["price"]
                .describe(percentiles=[0.25, 0.5, 0.75])
                .reset_index()
            )

            fig_sh = go.Figure()
            for _, row in pctile_df.iterrows():
                label = row["superhost_label"]
                color = C["accent"] if label == "Superhost" else C["neutral"]

                fig_sh.add_trace(
                    go.Bar(
                        name=label,
                        x=[label],
                        y=[row["50%"]],
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[row["75%"] - row["50%"]],
                            arrayminus=[row["50%"] - row["25%"]],
                        ),
                        marker_color=color,
                        text=f"€{row['50%']:.0f}",
                        textposition="outside",
                    )
                )

            fig_sh.update_layout(
                title="Superhost vs Regular Host — Median Price (IQR bars)",
                yaxis_title="Nightly price (€)",
                showlegend=False,
                bargap=0.5,
            )
            safe_plot(fig_sh, 420)
        else:
            st.info("No superhost data available.")

    st.markdown("---")

    sh_rev = (
        df.dropna(subset=["superhost_label"])
        .groupby("superhost_label")["reviews_per_month"]
        .median()
        .reset_index()
    )

    if not sh_rev.empty:
        fig_rev = px.bar(
            sh_rev,
            x="superhost_label",
            y="reviews_per_month",
            title="Median Review Velocity — Superhost vs Regular",
            labels={"superhost_label": "", "reviews_per_month": "Median reviews / month"},
            color="superhost_label",
            color_discrete_map={"Superhost": C["accent"], "Regular host": C["neutral"]},
            text="reviews_per_month",
        )
        fig_rev.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_rev.update_layout(showlegend=False)
        col_c, _, _ = st.columns([1, 1, 1])
        with col_c:
            safe_plot(fig_rev, 340)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — HOST CONCENTRATION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-header">Host Concentration Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
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

        if not hc_mix.empty:
            totals = hc_mix.groupby("neighbourhood_cleansed")["count"].transform("sum")
            hc_mix["pct"] = np.where(totals > 0, hc_mix["count"] / totals * 100, 0)

            multi = hc_mix[hc_mix["host_type"] == "Multi-listing"][["neighbourhood_cleansed", "pct"]]
            order = multi.sort_values("pct")["neighbourhood_cleansed"].tolist()
            if not order:
                order = hc_mix["neighbourhood_cleansed"].drop_duplicates().tolist()

            fig_hc = px.bar(
                hc_mix,
                x="pct",
                y="neighbourhood_cleansed",
                color="host_type",
                orientation="h",
                barmode="stack",
                category_orders={"neighbourhood_cleansed": order},
                title="Single vs Multi-Listing Host Share",
                labels={"pct": "Share (%)", "neighbourhood_cleansed": ""},
                color_discrete_map={
                    "Single-listing": C["neutral"],
                    "Multi-listing": C["secondary"],
                },
            )
            safe_plot(fig_hc, 480)
        else:
            st.info("No host concentration data available.")

    with c2:
        hc_price = (
            df.groupby("host_type")["price"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .reset_index()
        )

        if not hc_price.empty:
            fig_ht = go.Figure()

            for _, row in hc_price.iterrows():
                ht = row["host_type"]
                color = C["secondary"] if ht == "Multi-listing" else C["neutral"]

                fig_ht.add_trace(
                    go.Box(
                        name=ht,
                        q1=[row["25%"]],
                        median=[row["50%"]],
                        q3=[row["75%"]],
                        lowerfence=[row["min"]],
                        upperfence=[row["max"]],
                        marker_color=color,
                        line_width=2,
                    )
                )

            fig_ht.update_layout(
                title="Price Distribution — Single vs Multi-Listing Hosts",
                yaxis_title="Nightly price (€)",
                showlegend=True,
            )
            safe_plot(fig_ht, 480)
        else:
            st.info("No host-type price data available.")

        hc_hist = (
            df["host_listing_count_actual"]
            .fillna(1)
            .clip(upper=30)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        hc_hist.columns = ["listings_per_host", "host_count"]

        if not hc_hist.empty:
            fig_hh = px.bar(
                hc_hist,
                x="listings_per_host",
                y="host_count",
                title="Listings per Host (capped at 30)",
                labels={"listings_per_host": "Number of listings", "host_count": "Number of hosts"},
                color_discrete_sequence=[C["primary"]],
            )
            safe_plot(fig_hh, 300)
        else:
            st.info("No host distribution data available.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHT SHORTLIST
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        '<p class="section-header">Final Insight Shortlist for Notebook Transfer</p>',
        unsafe_allow_html=True,
    )

    insights = [
        {
            "id": "1",
            "title": "Neighbourhood price hierarchy is steep and non-linear",
            "finding": "The most expensive neighbourhoods sit well above the city median, while the gap narrows across the next tier.",
            "action": "Benchmark pricing separately for premium and mid-market neighbourhoods.",
            "chart": "Horizontal bar — top neighbourhoods by median price",
            "strength": "🔴 High",
        },
        {
            "id": "2",
            "title": "Entire homes command a clear premium",
            "finding": "Entire-home listings are priced above private rooms and show wider upside variation.",
            "action": "Treat full-property listings as the strongest revenue segment.",
            "chart": "Box plot — price by room type",
            "strength": "🔴 High",
        },
        {
            "id": "3",
            "title": "Superhost status aligns with stronger review velocity",
            "finding": "Superhosts tend to receive more reviews per month, suggesting better performance.",
            "action": "Improve host quality and listing optimisation to move toward superhost status.",
            "chart": "Bar chart — review velocity by host type",
            "strength": "🟠 Medium-High",
        },
        {
            "id": "4",
            "title": "Professional operators cluster in specific neighbourhoods",
            "finding": "Some neighbourhoods show a higher share of multi-listing hosts, pointing to stronger competitive pressure.",
            "action": "Treat those areas as more professionalised market segments.",
            "chart": "Stacked horizontal bar — host mix by neighbourhood",
            "strength": "🟠 Medium-High",
        },
        {
            "id": "5",
            "title": "High availability with weak reviews may signal underperformance",
            "finding": "Listings that remain widely available but attract few reviews may be overpriced or poorly positioned.",
            "action": "Use availability and reviews together as a performance signal.",
            "chart": "Scatter — availability vs reviews/month",
            "strength": "🟠 Medium",
        },
        {
            "id": "6",
            "title": "Guest capacity predicts price up to mid-sized listings",
            "finding": "Median price tends to rise with guest capacity, especially across smaller and mid-sized properties.",
            "action": "Target 4–6 guest listings for a strong price-flexibility balance.",
            "chart": "Line chart — median price vs accommodates",
            "strength": "🟠 Medium",
        },
        {
            "id": "7",
            "title": "Minimum-night rules shape pricing and booking pattern",
            "finding": "Short stays may support more review activity, while longer minimums may lift nightly pricing.",
            "action": "Set stay rules based on neighbourhood and guest segment.",
            "chart": "Bar chart — median price by minimum-night bucket",
            "strength": "🟡 Medium",
        },
    ]

    for ins in insights:
        with st.expander(f"🔷 Insight {ins['id']} — {ins['title']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**📊 Finding:**  \n{ins['finding']}")
                st.markdown(f"**🎯 Business recommendation:**  \n{ins['action']}")
            with col2:
                st.markdown(f"**Chart type:**  \n`{ins['chart']}`")
                st.markdown(f"**Priority:**  \n{ins['strength']}")

    st.markdown("---")
    st.info(
        "Next step: move the strongest 5–7 insights into your notebook/report, "
        "and pair each chart with a short interpretation and business implication."
    )
