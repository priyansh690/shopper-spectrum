"""
🛒 Shopper Spectrum – Streamlit Application
Customer Segmentation & Product Recommendation Engine
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum | E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Base Reset ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Background ──────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0c1b 0%, #111827 50%, #0d1226 100%);
}

/* ─── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1430 0%, #1a1d3e 100%);
    border-right: 1px solid #2d3154;
}
[data-testid="stSidebar"] * { color: #c8cce8 !important; }

/* ─── Hero Header ─────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 40%, #db2777 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(124, 58, 237, 0.4);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.82);
    margin: 0;
}

/* ─── Module Cards ────────────────────────────────────────── */
.module-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}
.module-card:hover {
    border-color: rgba(124,58,237,0.5);
}

/* ─── Section Headers ─────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.4rem;
    font-weight: 700;
    color: white;
    margin-bottom: 1.4rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #4f46e5;
}

/* ─── Product Cards ───────────────────────────────────────── */
.product-card {
    background: linear-gradient(135deg, rgba(79,70,229,0.15) 0%, rgba(124,58,237,0.08) 100%);
    border: 1px solid rgba(79,70,229,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.25s ease;
}
.product-card:hover {
    border-color: #7c3aed;
    background: linear-gradient(135deg, rgba(79,70,229,0.25) 0%, rgba(124,58,237,0.15) 100%);
    transform: translateX(4px);
}
.product-rank {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.product-name {
    color: white;
    font-weight: 500;
    font-size: 0.95rem;
    flex: 1;
}
.product-score {
    background: rgba(79,70,229,0.3);
    color: #a5b4fc;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ─── Segment Badges ──────────────────────────────────────── */
.segment-badge-high-value {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 0.5rem 1.4rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    box-shadow: 0 6px 24px rgba(124,58,237,0.5);
}
.segment-badge-regular {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: white;
    padding: 0.5rem 1.4rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    box-shadow: 0 6px 24px rgba(37,99,235,0.5);
}
.segment-badge-occasional {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: white;
    padding: 0.5rem 1.4rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    box-shadow: 0 6px 24px rgba(217,119,6,0.5);
}
.segment-badge-at-risk {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    padding: 0.5rem 1.4rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    box-shadow: 0 6px 24px rgba(220,38,38,0.5);
}

/* ─── Metric Cards ────────────────────────────────────────── */
.metric-mini {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.metric-mini .label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-mini .value { color: white; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }

/* ─── Inputs ──────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ─── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.55) !important;
}

/* ─── Dividers ────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ─── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
}

/* ─── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1430; }
::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MODEL LOADING                             ║
# ╚══════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all saved model artefacts."""
    required_models = [
        'kmeans_model.pkl',
        'scaler.pkl',
        'cluster_labels_map.pkl',
        'product_similarity.pkl',
        'product_names.pkl',
    ]
    
    models_paths = {}
    missing = []
    
    for model_file in required_models:
        if os.path.exists(model_file):
            models_paths[model_file] = model_file
        elif os.path.exists(os.path.join('models', model_file)):
            models_paths[model_file] = os.path.join('models', model_file)
        else:
            missing.append(model_file)
            
    if missing:
        st.error(f"Failed to find models. Missing: {missing}")
        st.stop()

    kmeans      = joblib.load(models_paths['kmeans_model.pkl'])
    scaler      = joblib.load(models_paths['scaler.pkl'])
    labels_map  = joblib.load(models_paths['cluster_labels_map.pkl'])
    sim_df      = joblib.load(models_paths['product_similarity.pkl'])
    prod_names  = joblib.load(models_paths['product_names.pkl'])
    return (kmeans, scaler, labels_map, sim_df, prod_names), []


@st.cache_data(show_spinner=False)
def load_rfm_data():
    path = 'rfm_segments.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback to data folder if it exists
    if os.path.exists('data/rfm_segments.csv'):
        return pd.read_csv('data/rfm_segments.csv')
    return None


# ╔══════════════════════════════════════════════════════════════╗
# ║                    HELPER FUNCTIONS                          ║
# ╚══════════════════════════════════════════════════════════════╝

def get_segment_info(segment: str) -> dict:
    info = {
        'High-Value': {
            'icon': '👑',
            'badge_class': 'segment-badge-high-value',
            'color': '#7c3aed',
            'description': 'Your most valuable customers. They purchase frequently, spend generously, and bought recently. Reward them with VIP perks and exclusive early access.',
            'actions': ['🎁 Offer loyalty rewards & VIP membership', '📧 Send personalised product previews', '🚀 Early access to new collections'],
        },
        'Regular': {
            'icon': '⭐',
            'badge_class': 'segment-badge-regular',
            'color': '#2563eb',
            'description': 'Steady, reliable customers with moderate spend. They respond well to cross-sell and upsell campaigns.',
            'actions': ['📦 Bundle product recommendations', '💌 Monthly newsletter with curated picks', '🏷️ Moderate discount offers to increase spend'],
        },
        'Occasional': {
            'icon': '🔔',
            'badge_class': 'segment-badge-occasional',
            'color': '#d97706',
            'description': 'Infrequent buyers who purchase sporadically. Seasonal promotions and re-engagement campaigns work well.',
            'actions': ['🎉 Seasonal sale promotions', '📱 Push notification for new arrivals', '💡 "We miss you" re-engagement emails'],
        },
        'At-Risk': {
            'icon': '⚠️',
            'badge_class': 'segment-badge-at-risk',
            'color': '#dc2626',
            'description': 'Customers who haven\'t purchased in a long time. Immediate win-back campaigns are critical to retain them.',
            'actions': ['💰 Win-back offer with steep discount', '📞 Personal outreach for high-value at-risk', '🆓 Free shipping incentive to re-engage'],
        },
    }
    return info.get(segment, info['Occasional'])


def get_recommendations(sim_df: pd.DataFrame, product_name: str, top_n: int = 5):
    product_name = product_name.strip().upper()
    if product_name in sim_df.index:
        scores = sim_df[product_name].drop(product_name).sort_values(ascending=False)
        matched = product_name
    else:
        matches = [p for p in sim_df.index if product_name in p]
        if not matches:
            return None, None
        matched = matches[0]
        scores = sim_df[matched].drop(matched).sort_values(ascending=False)
    return scores.head(top_n), matched


def gauge_chart(value: float, min_val: float, max_val: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'color': '#e2e8f0', 'size': 13}},
        number={'font': {'color': 'white', 'size': 22}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': '#64748b'},
            'bar': {'color': color},
            'bgcolor': '#1e293b',
            'bordercolor': '#334155',
            'steps': [
                {'range': [min_val, max_val * 0.33], 'color': '#1e293b'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': '#293548'},
                {'range': [max_val * 0.66, max_val], 'color': '#334155'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ╔══════════════════════════════════════════════════════════════╗
# ║                       SIDEBAR                                ║
# ╚══════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem;">🛒</div>
        <div style="font-size:1.3rem; font-weight:800; color:white; margin-top:0.3rem;">
            Shopper Spectrum
        </div>
        <div style="font-size:0.8rem; color:#6b7280; margin-top:0.2rem;">
            E-Commerce Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "**Navigate**",
        ["🏠 Dashboard", "🔍 Product Recommender", "👤 Customer Segmentation"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#6b7280; padding: 0.5rem 0;">
        <b style="color:#9ca3af;">📌 Tech Stack</b><br>
        Python · Scikit-learn · Streamlit<br>
        KMeans · Cosine Similarity<br>
        RFM Analysis · Plotly
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     LOAD ASSETS                              ║
# ╚══════════════════════════════════════════════════════════════╝

models_tuple, missing_files = load_models()
rfm_data = load_rfm_data()

if missing_files:
    st.error(f"""
    ⚠️ **Model files not found.** Please run the Jupyter notebook first to generate all model files.

    Missing: `{', '.join(missing_files)}`
    """)
    st.stop()

kmeans_model, scaler_model, cluster_labels_map, sim_df, product_names = models_tuple


# ╔══════════════════════════════════════════════════════════════╗
# ║                    PAGE: DASHBOARD                           ║
# ╚══════════════════════════════════════════════════════════════╝

if page == "🏠 Dashboard":

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🛒 Shopper Spectrum</h1>
        <p>Customer Segmentation & Product Recommendation Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    n_products = len(product_names)
    n_customers = len(rfm_data) if rfm_data is not None else "—"
    n_segments  = 4

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("🛍️", str(n_products), "Products Indexed"),
        ("👥", f"{n_customers:,}" if isinstance(n_customers, int) else n_customers, "Customers Analysed"),
        ("🎯", str(n_segments), "Customer Segments"),
        ("⚡", "Real-Time", "Predictions"),
    ]
    for col, (icon, val, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-mini">
                <div style="font-size:1.8rem;">{icon}</div>
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # RFM overview if data available
    if rfm_data is not None and 'Segment' in rfm_data.columns:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-header">📊 Segment Distribution</div>', unsafe_allow_html=True)
            seg_counts = rfm_data['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            colors_map = {
                'High-Value': '#7c3aed',
                'Regular':    '#2563eb',
                'Occasional': '#d97706',
                'At-Risk':    '#dc2626',
            }
            fig_pie = px.pie(
                seg_counts,
                names='Segment',
                values='Count',
                color='Segment',
                color_discrete_map=colors_map,
                template='plotly_dark',
                hole=0.55,
            )
            fig_pie.update_traces(
                textinfo='label+percent',
                textfont_size=12,
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(font=dict(color='white')),
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">💡 Segment KPIs</div>', unsafe_allow_html=True)
            seg_kpi = rfm_data.groupby('Segment').agg(
                Customers=('CustomerID', 'count'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Frequency=('Frequency', 'mean'),
                Avg_Monetary=('Monetary', 'mean'),
            ).round(1).reset_index()

            fig_bar = go.Figure()
            for metric, color in [('Avg_Recency', '#7c3aed'), ('Avg_Frequency', '#2563eb'), ('Avg_Monetary', '#059669')]:
                fig_bar.add_trace(go.Bar(
                    name=metric.replace('Avg_', 'Avg '),
                    x=seg_kpi['Segment'],
                    y=seg_kpi[metric] / seg_kpi[metric].max(),
                    marker_color=color,
                    opacity=0.85,
                ))
            fig_bar.update_layout(
                barmode='group',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                yaxis_title='Normalised Value',
                legend=dict(font=dict(color='white')),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Table
        st.markdown('<div class="section-header">📋 Detailed Segment Statistics</div>', unsafe_allow_html=True)
        display_kpi = seg_kpi.copy()
        display_kpi.columns = ['Segment', 'Customers', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary (£)']
        st.dataframe(
            display_kpi.set_index('Segment').style
            .format({'Avg Recency (days)': '{:.1f}', 'Avg Frequency': '{:.1f}', 'Avg Monetary (£)': '£{:,.2f}', 'Customers': '{:,}'}),
            use_container_width=True,
        )

    # Module cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 Modules</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="module-card">
            <div style="font-size:2.2rem; margin-bottom:0.7rem;">🔍</div>
            <div style="color:white; font-size:1.1rem; font-weight:700; margin-bottom:0.5rem;">Product Recommender</div>
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.5;">
                Enter any product name and instantly discover 5 similar products using
                item-based collaborative filtering with cosine similarity.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="module-card">
            <div style="font-size:2.2rem; margin-bottom:0.7rem;">👤</div>
            <div style="color:white; font-size:1.1rem; font-weight:700; margin-bottom:0.5rem;">Customer Segmentation</div>
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.5;">
                Input customer RFM metrics to predict their segment: High-Value, Regular,
                Occasional, or At-Risk — with actionable retention strategies.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║               PAGE: PRODUCT RECOMMENDER                      ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "🔍 Product Recommender":

    st.markdown("""
    <div class="hero">
        <h1>🔍 Product Recommender</h1>
        <p>Item-based Collaborative Filtering · Cosine Similarity · Real-Time</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_info = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('<div class="section-header">✏️ Find Similar Products</div>', unsafe_allow_html=True)

        # Autocomplete hint
        st.markdown(f"<div style='color:#6b7280; font-size:0.85rem; margin-bottom:0.5rem;'>💡 {len(product_names)} products indexed. Type a product keyword to search.</div>", unsafe_allow_html=True)

        product_input = st.text_input(
            "Product Name",
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
            label_visibility="collapsed",
        )

        # Show matching suggestions
        if product_input and len(product_input) >= 3:
            matches = [p for p in product_names if product_input.upper() in p][:8]
            if matches:
                st.markdown("<div style='color:#94a3b8; font-size:0.82rem; margin:0.3rem 0 0.5rem;'>Matching products:</div>", unsafe_allow_html=True)
                selected = st.selectbox("Select a product", ["— type or select below —"] + matches, label_visibility="collapsed")
                if selected != "— type or select below —":
                    product_input = selected

        top_n = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5, step=1)
        get_btn = st.button("🚀 Get Recommendations", key="rec_btn")

    with col_info:
        st.markdown('<div class="section-header">ℹ️ How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="module-card">
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
                <b style="color:white;">1.</b> Build a <b style="color:#a5b4fc;">Customer × Product</b> matrix from transaction history.<br>
                <b style="color:white;">2.</b> Compute <b style="color:#a5b4fc;">Cosine Similarity</b> between each pair of products.<br>
                <b style="color:white;">3.</b> For a given product, return the top-N most similar items.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Results
    if get_btn or (product_input and product_input.strip()):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            recs, matched = get_recommendations(sim_df, product_input, top_n)
            st.markdown("---")

            if recs is None:
                st.error(f"❌ No product matching **\"{product_input}\"** found. Try a different keyword.")
            else:
                col_res1, col_res2 = st.columns([3, 2], gap="large")

                with col_res1:
                    st.markdown(f"""
                    <div style="margin-bottom:1rem;">
                        <div style="color:#94a3b8; font-size:0.85rem;">Showing recommendations for:</div>
                        <div style="color:white; font-size:1.1rem; font-weight:700; margin-top:0.2rem;">
                            🛍️ {matched}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-header">🎯 Recommended Products</div>', unsafe_allow_html=True)
                    for i, (product, score) in enumerate(recs.items(), 1):
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="product-rank">{i}</div>
                            <div class="product-name">{product}</div>
                            <div class="product-score">{score:.1%} match</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_res2:
                    st.markdown('<div class="section-header">📊 Similarity Scores</div>', unsafe_allow_html=True)
                    rec_df = pd.DataFrame({
                        'Product': [p[:30] + '…' if len(p) > 30 else p for p in recs.index],
                        'Score': recs.values,
                    })
                    fig_h = go.Figure(go.Bar(
                        x=rec_df['Score'],
                        y=rec_df['Product'],
                        orientation='h',
                        marker=dict(
                            color=rec_df['Score'],
                            colorscale=[[0, '#312e81'], [1, '#7c3aed']],
                            showscale=False,
                        ),
                        text=[f"{s:.1%}" for s in rec_df['Score']],
                        textposition='outside',
                        textfont=dict(color='white', size=11),
                    ))
                    fig_h.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=350,
                        xaxis=dict(range=[0, 1.1], title='Cosine Similarity'),
                        yaxis=dict(autorange='reversed'),
                        margin=dict(l=0, r=60, t=10, b=10),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

    # Browse products
    with st.expander("📋 Browse All Indexed Products"):
        search_browse = st.text_input("Filter", placeholder="Search products …", key="browse_search")
        filtered = [p for p in product_names if search_browse.upper() in p] if search_browse else product_names
        st.dataframe(
            pd.DataFrame({'Product Name': filtered}),
            use_container_width=True,
            height=300,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║             PAGE: CUSTOMER SEGMENTATION                      ║
# ╚══════════════════════════════════════════════════════════════╝

elif page == "👤 Customer Segmentation":

    st.markdown("""
    <div class="hero">
        <h1>👤 Customer Segmentation</h1>
        <p>RFM-Based KMeans Clustering · Predict Customer Segment in Real-Time</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">📥 Enter Customer RFM Values</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div class="module-card">
            """, unsafe_allow_html=True)

            recency   = st.number_input("📅 Recency (days since last purchase)",
                                        min_value=0, max_value=1000, value=30, step=1,
                                        help="Number of days since the customer last made a purchase.")
            frequency = st.number_input("🔁 Frequency (number of transactions)",
                                        min_value=1, max_value=500, value=10, step=1,
                                        help="Total number of distinct invoices by this customer.")
            monetary  = st.number_input("💰 Monetary (total spend in £)",
                                        min_value=0.0, max_value=100000.0, value=500.0, step=10.0,
                                        format="%.2f",
                                        help="Total revenue generated by this customer.")

            st.markdown("</div>", unsafe_allow_html=True)

            predict_btn = st.button("🎯 Predict Customer Segment", key="seg_btn")

        # Reference table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📖 Segment Reference</div>', unsafe_allow_html=True)
        ref_data = {
            'Segment':   ['High-Value', 'Regular',     'Occasional',  'At-Risk'],
            'Recency':   ['Low (recent)','Moderate',   'High',        'Very High'],
            'Frequency': ['High',        'Medium',     'Low',         'Very Low'],
            'Monetary':  ['High',        'Medium',     'Low',         'Low'],
        }
        st.dataframe(pd.DataFrame(ref_data).set_index('Segment'), use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Scale & predict
            rfm_input  = np.array([[recency, frequency, monetary]])
            rfm_scaled = scaler_model.transform(rfm_input)
            cluster_id = kmeans_model.predict(rfm_scaled)[0]
            segment    = cluster_labels_map.get(cluster_id, 'Unknown')
            info       = get_segment_info(segment)

            # Segment badge
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:1.5rem;">
                <div style="font-size:3.5rem; margin-bottom:0.5rem;">{info['icon']}</div>
                <span class="{info['badge_class']}">{segment} Customer</span>
            </div>
            """, unsafe_allow_html=True)

            # Description
            st.markdown(f"""
            <div class="module-card" style="margin-bottom:1.2rem;">
                <div style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                    {info['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Input summary
            c1, c2, c3 = st.columns(3)
            for col, label, val, unit in zip(
                [c1, c2, c3],
                ['Recency', 'Frequency', 'Monetary'],
                [recency, frequency, monetary],
                ['days', 'orders', '£'],
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-mini">
                        <div class="label">{label}</div>
                        <div class="value">{val:,.0f}</div>
                        <div style="color:#6b7280; font-size:0.75rem;">{unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge charts
            if rfm_data is not None:
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    st.plotly_chart(gauge_chart(recency, 0, int(rfm_data['Recency'].max()),
                                                'Recency (days)', '#7c3aed'), use_container_width=True)
                with gc2:
                    st.plotly_chart(gauge_chart(frequency, 0, int(rfm_data['Frequency'].max()),
                                                'Frequency', '#2563eb'), use_container_width=True)
                with gc3:
                    st.plotly_chart(gauge_chart(monetary, 0, float(rfm_data['Monetary'].quantile(0.99)),
                                                'Monetary (£)', '#059669'), use_container_width=True)

            # Recommended actions
            st.markdown('<div class="section-header" style="margin-top:1rem;">💼 Recommended Actions</div>', unsafe_allow_html=True)
            for action in info['actions']:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px; padding:0.7rem 1rem; margin:0.4rem 0; color:#e2e8f0; font-size:0.9rem;">
                    {action}
                </div>
                """, unsafe_allow_html=True)

            # Cluster position in context of all customers
            if rfm_data is not None and 'Segment' in rfm_data.columns:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">📍 Where This Customer Sits</div>', unsafe_allow_html=True)
                sample = rfm_data.sample(min(1500, len(rfm_data)), random_state=42)
                seg_color_map = {
                    'High-Value': '#7c3aed', 'Regular': '#2563eb',
                    'Occasional': '#d97706', 'At-Risk': '#dc2626',
                }
                fig_scatter = px.scatter(
                    sample,
                    x='Recency', y='Monetary',
                    color='Segment',
                    color_discrete_map=seg_color_map,
                    size='Frequency',
                    size_max=12,
                    opacity=0.5,
                    template='plotly_dark',
                    height=380,
                )
                # Highlight the input point
                fig_scatter.add_trace(go.Scatter(
                    x=[recency], y=[monetary],
                    mode='markers+text',
                    marker=dict(size=18, color='white', symbol='star',
                                line=dict(color=info['color'], width=3)),
                    text=['You'], textposition='top center',
                    textfont=dict(color='white', size=12),
                    name='Input Customer',
                    showlegend=True,
                ))
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(font=dict(color='white')),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            # Placeholder state
            st.markdown("""
            <div style="text-align:center; padding:4rem 1rem; color:#4b5563;">
                <div style="font-size:4rem; margin-bottom:1rem;">🎯</div>
                <div style="font-size:1rem;">Enter customer RFM values and click<br>
                <b style="color:#6b7280;">Predict Customer Segment</b> to see results.</div>
            </div>
            """, unsafe_allow_html=True)
