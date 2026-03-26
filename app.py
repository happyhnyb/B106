"""
Berlin Airbnb Market Analysis — Streamlit Dashboard
B106 Data Visualisation | Inside Airbnb Dataset
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Berlin Airbnb Market Analysis",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "primary"   : "#2E5D9E",
    "secondary" : "#E8834E",
    "accent"    : "#4CAF82",
    "neutral"   : "#8C9EB0",
    "light"     : "#F5F7FA",
    "dark"      : "#1C2B3A",
    "bg"        : "#FFFFFF",
    "card_bg"   : "#F0F4FA",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, sans-serif", color=C["dark"]),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["light"],
        colorway=[C["primary"], C["secondary"], C["accent"], C["neutral"],
                  "#9B5FBF", "#E5C84A", "#3DB5B5"],
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
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* KPI cards */
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

  /* Section headers */
  .section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1C2B3A;
    border-bottom: 2px solid #E8EEF4;
    padding-bottom: 8px;
    margin: 16px 0 12px 0;
  }

  /* Insight badge */
  .insight-badge {
    display: inline-block;
    background: #2E5D9E;
    color: white;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 20px;
    padding: 2px 10px;
    margin-right: 6px;
    vertical-align: middle;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1C2B3A 0%, #2E3D50 100%);
  }
  [data-testid="stSidebar"] * { color: #E0E8F0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label { color: #A8BDD0 !important; font-size: 0.82rem; }

  /* Plotly chart container */
  .chart-container {
    background: white;
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 12px;
  }

  /* Streamlit overrides */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 0.88rem;
  }
  div[data-testid="stMetricValue"] > div { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(path: str) -> pd.DataFrame:
    """Load, clean, and engineer features from the raw listings file."""
    df = pd.read_csv(path, low_memory=False)

    # Price
    df["price"] = (
        df["price"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("nan", np.nan)
        .astype(float)
    )

    # Outlier trim
    q1, q99 = df["price"].quantile([0.01, 0.99])
    df = df[(df["price"] >= q1) & (df["price"] <= q99)].copy()

    # Booleans
    for col in ["host_is_superhost", "host_has_profile_pic",
                "host_identity_verified", "instant_bookable"]:
        if col in df.columns:
            df[col] = df[col].map({"t": True, "f": False})

    # Dates
    for col in ["last_review", "host_since"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Rate strings
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace("%", "")
                .replace("nan", np.nan).astype(float)
            )

    # Derived
    hc = df.groupby("host_id")["id"].count()
    df["host_listing_count_actual"] = df["host_id"].map(hc)
    df["host_type"] = df["host_listing_count_actual"].apply(
        lambda x: "Multi-listing" if x > 1 else "Single-listing"
    )

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    def nights_bucket(n):
        if   n <= 1  : return "1 night"
        elif n <= 3  : return "2–3 nights"
        elif n <= 7  : return "4–7 nights"
        elif n <= 30 : return "8–30 nights"
        else         : return "30+ nights"

    df["nights_bucket"] = df["minimum_nights"].apply(nights_bucket)
    df["superhost_label"] = df["host_is_superhost"].map(
        {True: "Superhost", False: "Regular host"}
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Berlin Airbnb")
    st.markdown("**Market Analysis Dashboard**")
    st.markdown("---")

    DATA_PATH = st.text_input(
        "Dataset path",
        value="data/listings.csv.gz",
        help="Relative path to listings.csv.gz from insideairbnb.com",
    )

    st.markdown("### Filters")

    try:
        with st.spinner("Loading data…"):
            df_full = load_and_clean(DATA_PATH)
        data_ok = True
    except FileNotFoundError:
        st.error(
            "⚠️ Data file not found.\n\n"
            "Download `listings.csv.gz` from **insideairbnb.com/get-the-data** "
            "and place it in the `data/` folder."
        )
        data_ok = False
        df_full = pd.DataFrame()

    if data_ok:
        all_neighbourhoods = sorted(df_full["neighbourhood_cleansed"].dropna().unique())
        all_room_types     = sorted(df_full["room_type"].dropna().unique())
        price_min, price_max = float(df_full["price"].min()), float(df_full["price"].max())

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
            min_value=int(price_min),
            max_value=int(price_max),
            value=(int(price_min), int(price_max)),
            step=5,
        )
        superhost_filter = st.selectbox(
            "Host type",
            ["All", "Superhost only", "Regular hosts only"],
        )

        st.markdown("---")
        st.caption("Dataset: [Inside Airbnb Berlin](https://insideairbnb.com/berlin/)")
        st.caption("Course: B106 Data Visualisation")


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])
    if selected_neighbourhoods:
        mask &= df["neighbourhood_cleansed"].isin(selected_neighbourhoods)
    if selected_room_types:
        mask &= df["room_type"].isin(selected_room_types)
    if superhost_filter == "Superhost only":
        mask &= df["host_is_superhost"] == True
    elif superhost_filter == "Regular hosts only":
        mask &= df["host_is_superhost"] == False
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: render a Plotly chart with our template
# ─────────────────────────────────────────────────────────────────────────────
def render(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=height)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if not data_ok:
    st.title("🏙️ Berlin Airbnb Market Analysis")
    st.info("👈 Add your dataset path in the sidebar to begin.")
    st.stop()

df = apply_filters(df_full)

st.title("🏙️ Berlin Short-Term Rental Market")
st.markdown(
    f"Analysing **{len(df):,}** listings "
    f"{'(filtered from ' + str(len(df_full):,) + ')' if len(df) < len(df_full) else ''} "
    "· Inside Airbnb Berlin dataset"
)

# ── TABS ──────────────────────────────────────────────────────────────────────
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

    kpi_data = [
        ("Total Listings",        f"{len(df):,}",                           "Active on Airbnb"),
        ("Median Price",          f"€{df['price'].median():.0f}",           "Per night"),
        ("Superhost Share",       f"{df['host_is_superhost'].mean()*100:.1f}%" if df['host_is_superhost'].notna().any() else "N/A", "Of all hosts"),
        ("Unique Neighbourhoods", f"{df['neighbourhood_cleansed'].nunique():,}", "Across Berlin"),
        ("Avg Reviews/Month",     f"{df['reviews_per_month'].mean():.2f}",  "Proxy for bookings"),
        ("Multi-listing Hosts",   f"{(df['host_type']=='Multi-listing').mean()*100:.1f}%", "Professional operators"),
    ]

    cols = st.columns(6)
    for col, (label, value, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-delta">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        fig_hist = px.histogram(
            df, x="price", nbins=60,
            title="Nightly Price Distribution",
            labels={"price": "Nightly price (€)", "count": "Listings"},
            color_discrete_sequence=[C["primary"]],
        )
        fig_hist.update_traces(marker_line_width=0)
        fig_hist.add_vline(
            x=df["price"].median(), line_dash="dash",
            line_color=C["secondary"], line_width=2,
            annotation_text=f"Median: €{df['price'].median():.0f}",
            annotation_position="top right",
        )
        st.plotly_chart(render(fig_hist, 360), use_container_width=True)

    with c2:
        room_counts = df["room_type"].value_counts().reset_index()
        room_counts.columns = ["room_type", "count"]
        fig_pie = px.pie(
            room_counts, names="room_type", values="count",
            title="Listing Share by Room Type",
            color_discrete_sequence=[C["primary"], C["secondary"],
                                      C["accent"], C["neutral"]],
            hole=0.42,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(render(fig_pie, 360), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — NEIGHBOURHOODS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-header">Neighbourhood Analysis</p>', unsafe_allow_html=True)

    neigh_agg = (
        df.groupby("neighbourhood_cleansed")
        .agg(
            median_price=("price", "median"),
            listing_count=("id", "count"),
            avg_reviews=("reviews_per_month", "mean"),
        )
        .reset_index()
        .sort_values("median_price", ascending=False)
    )

    top_n = st.slider("Top N neighbourhoods to show", 5, 30, 15)
    top_neigh = neigh_agg.head(top_n)

    fig_neigh = px.bar(
        top_neigh.sort_values("median_price"),
        x="median_price", y="neighbourhood_cleansed",
        orientation="h",
        title=f"Top {top_n} Neighbourhoods by Median Nightly Price",
        labels={
            "median_price": "Median price (€)",
            "neighbourhood_cleansed": "",
        },
        color="median_price",
        color_continuous_scale=[[0, "#A8C4E0"], [1, C["primary"]]],
        text="median_price",
    )
    fig_neigh.update_traces(
        texttemplate="€%{text:.0f}", textposition="outside"
    )
    fig_neigh.update_coloraxes(showscale=False)
    st.plotly_chart(render(fig_neigh, 520), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig_bubble = px.scatter(
            neigh_agg.head(20),
            x="median_price", y="avg_reviews",
            size="listing_count",
            text="neighbourhood_cleansed",
            title="Price vs Review Velocity — Top 20 Neighbourhoods",
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
        st.plotly_chart(render(fig_bubble, 420), use_container_width=True)

    with c2:
        # Room type breakdown per neighbourhood
        rt_neigh = (
            df[df["neighbourhood_cleansed"].isin(neigh_agg.head(12)["neighbourhood_cleansed"])]
            .groupby(["neighbourhood_cleansed", "room_type"])
            .size()
            .reset_index(name="count")
        )
        total_per_neigh = rt_neigh.groupby("neighbourhood_cleansed")["count"].transform("sum")
        rt_neigh["pct"] = rt_neigh["count"] / total_per_neigh * 100

        fig_rt = px.bar(
            rt_neigh,
            x="pct", y="neighbourhood_cleansed",
            color="room_type",
            orientation="h",
            barmode="stack",
            title="Room Type Mix — Top 12 Neighbourhoods",
            labels={"pct": "Share (%)", "neighbourhood_cleansed": ""},
            color_discrete_sequence=[C["primary"], C["secondary"],
                                      C["accent"], C["neutral"]],
        )
        st.plotly_chart(render(fig_rt, 420), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — LISTING CHARACTERISTICS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header">Listing Characteristics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Price by room type — separate box per type
        fig_box = px.box(
            df,
            x="room_type", y="price",
            color="room_type",
            title="Price Distribution by Room Type",
            labels={"price": "Nightly price (€)", "room_type": ""},
            color_discrete_sequence=[C["primary"], C["secondary"],
                                      C["accent"], C["neutral"]],
            points="outliers",
        )
        fig_box.update_traces(boxmean=True)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(render(fig_box, 400), use_container_width=True)

    with c2:
        if "accommodates" in df.columns:
            cap_agg = (
                df[df["accommodates"].between(1, 10)]
                .groupby("accommodates")["price"]
                .median()
                .reset_index()
            )
            fig_cap = px.line(
                cap_agg,
                x="accommodates", y="price",
                title="Median Price by Guest Capacity",
                labels={"accommodates": "Guests accommodated", "price": "Median price (€)"},
                markers=True,
                color_discrete_sequence=[C["primary"]],
            )
            fig_cap.update_traces(
                marker_size=9,
                marker_color=C["secondary"],
                line_width=2.5,
            )
            st.plotly_chart(render(fig_cap, 400), use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        mn_order = ["1 night", "2–3 nights", "4–7 nights", "8–30 nights", "30+ nights"]
        mn_agg = (
            df.groupby("nights_bucket")
            .agg(median_price=("price", "median"), count=("id", "count"))
            .reindex(mn_order)
            .dropna()
            .reset_index()
        )
        fig_mn = px.bar(
            mn_agg,
            x="nights_bucket", y="median_price",
            title="Median Price by Minimum-Night Policy",
            labels={"nights_bucket": "Minimum nights", "median_price": "Median price (€)"},
            color="median_price",
            color_continuous_scale=[[0, "#A8C4E0"], [1, C["primary"]]],
            text="median_price",
        )
        fig_mn.update_traces(texttemplate="€%{text:.0f}", textposition="outside")
        fig_mn.update_coloraxes(showscale=False)
        st.plotly_chart(render(fig_mn, 380), use_container_width=True)

    with c4:
        if "review_scores_rating" in df.columns:
            rat_df = df.dropna(subset=["review_scores_rating"])
            fig_rat = px.histogram(
                rat_df, x="review_scores_rating",
                nbins=30,
                title="Distribution of Review Scores",
                labels={"review_scores_rating": "Review score", "count": "Listings"},
                color_discrete_sequence=[C["accent"]],
            )
            st.plotly_chart(render(fig_rat, 380), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRICING & DEMAND
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header">Pricing & Demand Signals</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        avail_col = next(
            (c for c in ["availability_365", "availability_90", "availability_30"]
             if c in df.columns), None
        )
        if avail_col:
            avail_df = df[[avail_col, "reviews_per_month", "room_type"]].dropna()
            avail_df = avail_df[avail_df[avail_col] > 0]
            if len(avail_df) > 3000:
                avail_df = avail_df.sample(3000, random_state=42)
            fig_sc = px.scatter(
                avail_df,
                x=avail_col, y="reviews_per_month",
                color="room_type",
                opacity=0.35,
                title="Availability vs Review Velocity",
                labels={
                    avail_col: "Days available / year",
                    "reviews_per_month": "Reviews / month",
                    "room_type": "Room type",
                },
                trendline="lowess",
                color_discrete_sequence=[C["primary"], C["secondary"],
                                          C["accent"], C["neutral"]],
            )
            fig_sc.update_traces(marker_size=5)
            st.plotly_chart(render(fig_sc, 420), use_container_width=True)

    with c2:
        # Price percentile by superhost
        sh_df = df.dropna(subset=["superhost_label"])
        pctile_df = (
            sh_df.groupby("superhost_label")["price"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .reset_index()
        )
        fig_sh = go.Figure()
        for _, row in pctile_df.iterrows():
            label = row["superhost_label"]
            color = C["accent"] if label == "Superhost" else C["neutral"]
            fig_sh.add_trace(go.Bar(
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
                text=f'€{row["50%"]:.0f}',
                textposition="outside",
            ))
        fig_sh.update_layout(
            title="Superhost vs Regular Host — Median Price (IQR bars)",
            yaxis_title="Nightly price (€)",
            showlegend=False,
            bargap=0.5,
        )
        st.plotly_chart(render(fig_sh, 420), use_container_width=True)

    st.markdown("---")
    # Review velocity comparison
    sh_rev = (
        df.dropna(subset=["superhost_label"])
        .groupby("superhost_label")["reviews_per_month"]
        .median()
        .reset_index()
    )
    fig_rev = px.bar(
        sh_rev,
        x="superhost_label", y="reviews_per_month",
        title="Median Review Velocity — Superhost vs Regular",
        labels={"superhost_label": "", "reviews_per_month": "Median reviews / month"},
        color="superhost_label",
        color_discrete_map={"Superhost": C["accent"], "Regular host": C["neutral"]},
        text="reviews_per_month",
    )
    fig_rev.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_rev.update_layout(showlegend=False, width=500)
    col_c, _, _ = st.columns([1, 1, 1])
    with col_c:
        st.plotly_chart(render(fig_rev, 340), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — HOST CONCENTRATION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-header">Host Concentration Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        top_n2 = st.slider("Top N neighbourhoods", 5, 20, 12, key="hc_slider")
        top_n_list = df["neighbourhood_cleansed"].value_counts().head(top_n2).index.tolist()
        df_hc = df[df["neighbourhood_cleansed"].isin(top_n_list)]

        hc_mix = (
            df_hc.groupby(["neighbourhood_cleansed", "host_type"])
            .size()
            .reset_index(name="count")
        )
        totals = hc_mix.groupby("neighbourhood_cleansed")["count"].transform("sum")
        hc_mix["pct"] = hc_mix["count"] / totals * 100

        order = (
            hc_mix[hc_mix["host_type"] == "Multi-listing"]
            .sort_values("pct")["neighbourhood_cleansed"]
            .tolist()
        )
        fig_hc = px.bar(
            hc_mix,
            x="pct", y="neighbourhood_cleansed",
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
        st.plotly_chart(render(fig_hc, 480), use_container_width=True)

    with c2:
        hc_price = (
            df.groupby("host_type")["price"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .reset_index()
        )
        fig_ht = go.Figure()
        for _, row in hc_price.iterrows():
            ht = row["host_type"]
            color = C["secondary"] if ht == "Multi-listing" else C["neutral"]
            fig_ht.add_trace(go.Box(
                name=ht,
                q1=[row["25%"]], median=[row["50%"]],
                q3=[row["75%"]], mean=[row["mean"]],
                lowerfence=[row["min"]], upperfence=[row["max"]],
                marker_color=color,
                line_width=2,
            ))
        fig_ht.update_layout(
            title="Price Distribution — Single vs Multi-Listing Hosts",
            yaxis_title="Nightly price (€)",
            showlegend=True,
        )
        st.plotly_chart(render(fig_ht, 480), use_container_width=True)

        # Host listing count histogram
        hc_hist = df["host_listing_count_actual"].clip(upper=30).value_counts().sort_index().reset_index()
        hc_hist.columns = ["listings_per_host", "host_count"]
        fig_hh = px.bar(
            hc_hist,
            x="listings_per_host", y="host_count",
            title="Listings per Host (capped at 30)",
            labels={"listings_per_host": "Number of listings", "host_count": "Number of hosts"},
            color_discrete_sequence=[C["primary"]],
        )
        st.plotly_chart(render(fig_hh, 300), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHT SHORTLIST
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-header">Final Insight Shortlist for Notebook Transfer</p>',
                unsafe_allow_html=True)

    insights = [
        {
            "id": "1",
            "title": "Neighbourhood price hierarchy is steep and non-linear",
            "finding": "The top-5 most expensive neighbourhoods command prices significantly above the city median, but the gap between #5 and #15 is narrower — implying diminishing marginal premium below the top tier.",
            "action": "Prioritise acquisition in the top-5 neighbourhoods. Model pricing benchmarks separately for tier-1 vs tier-2 areas.",
            "chart": "Horizontal bar — top 15 neighbourhoods by median price",
            "strength": "🔴 High",
        },
        {
            "id": "2",
            "title": "Entire homes command a clear price premium over private rooms",
            "finding": "The median price for entire-home listings is substantially higher than private rooms. The distributional spread within entire homes also shows greater upside potential.",
            "action": "Focus portfolio on entire-home contracts. Room-level subletting is unlikely to achieve competitive returns.",
            "chart": "Faceted box plot — one panel per room type",
            "strength": "🔴 High",
        },
        {
            "id": "3",
            "title": "Superhost status correlates with higher review velocity",
            "finding": "Superhost-designated listings accumulate more reviews per month than regular hosts, suggesting they secure more bookings even when prices are comparable.",
            "action": "Pursue superhost status for all managed properties as a long-term demand-generation strategy.",
            "chart": "Side-by-side bar charts — price and review velocity, separated",
            "strength": "🟠 Medium-High",
        },
        {
            "id": "4",
            "title": "Professional (multi-listing) hosts concentrate in specific neighbourhoods",
            "finding": "A subset of Berlin neighbourhoods shows disproportionately high multi-listing host share, signalling intense professional competition in those areas.",
            "action": "Avoid entering neighbourhoods with >50% multi-listing share without a strong differentiation strategy.",
            "chart": "Stacked horizontal bar — host type share by neighbourhood",
            "strength": "🟠 Medium-High",
        },
        {
            "id": "5",
            "title": "High availability correlates with weaker review activity",
            "finding": "Listings open for most of the year but accumulating few reviews may indicate overpricing, poor demand, or inadequate management — not strategic calendar management.",
            "action": "Use dynamic availability control. Long open calendars without bookings are a warning signal, not a feature.",
            "chart": "Scatter with LOWESS trend — availability vs reviews/month",
            "strength": "🟠 Medium",
        },
        {
            "id": "6",
            "title": "Guest capacity is a reliable price predictor up to 6 guests",
            "finding": "Median price rises consistently from 1 to 6 guests. Beyond 6, the marginal per-night premium per additional guest flattens.",
            "action": "Target 4–6 guest properties for optimal rate-per-sq-metre. Investing to increase capacity beyond 6 offers diminishing pricing returns.",
            "chart": "Line chart (single line) — median price vs accommodates",
            "strength": "🟠 Medium",
        },
        {
            "id": "7",
            "title": "Minimum-night policies expose a review–price trade-off",
            "finding": "Short-minimum listings generate more reviews but are not always higher priced; longer-minimum listings earn slightly more per night but fewer total reviews.",
            "action": "Apply 1–3 night minimums in tourist zones for volume/ranking; apply 7+ nights in corporate/residential areas for simplicity and per-night rate.",
            "chart": "Two separate bar charts — reviews/month and price by night bucket",
            "strength": "🟡 Medium",
        },
    ]

    for ins in insights:
        with st.expander(
            f"🔷 Insight {ins['id']} — {ins['title']}", expanded=False
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**📊 Finding:**  \n{ins['finding']}")
                st.markdown(f"**🎯 Business recommendation:**  \n{ins['action']}")
            with col2:
                st.markdown(f"**Chart type:**  \n`{ins['chart']}`")
                st.markdown(f"**Priority:**  \n{ins['strength']}")

    st.markdown("---")
    st.info(
        "**Next step:** Transfer the 🔴 High priority insights to the Jupyter Notebook first. "
        "Rebuild each chart in a clean notebook cell with explanatory markdown above and below. "
        "Aim for 6–7 insights total — quality over quantity."
    )
