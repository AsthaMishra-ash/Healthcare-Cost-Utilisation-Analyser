import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../data/cms_inpatient.csv')
MODELS = os.path.join(BASE, '../models')

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthcare Cost Analyser", page_icon="🏥", layout="wide")

# ── load assets ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_models():
    with open(os.path.join(MODELS, 'rf_model.pkl'),  'rb') as f: rf = pickle.load(f)
    with open(os.path.join(MODELS, 'lr_model.pkl'),  'rb') as f: lr = pickle.load(f)
    with open(os.path.join(MODELS, 'results.pkl'),   'rb') as f: results = pickle.load(f)
    with open(os.path.join(MODELS, 'encoders.pkl'),  'rb') as f: encoders = pickle.load(f)
    return rf, lr, results, encoders

df       = load_data()
rf, lr, results, encoders = load_models()

PALETTE = ['#1565C0','#1976D2','#42A5F5','#90CAF9','#BBDEFB','#E3F2FD']
CAT_COLORS = {
    'Cardiovascular': '#E53935', 'Respiratory': '#1E88E5',
    'Orthopedic':     '#43A047', 'Neurological': '#8E24AA',
    'Digestive':      '#FB8C00', 'Renal':        '#00897B',
    'Oncology':       '#6D4C41'
}

# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
st.sidebar.title("Healthcare Cost\nAnalyser")
st.sidebar.markdown("*CMS Medicare Inpatient Data*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Overview",
    " Regional Analysis",
    " Diagnosis Insights",
    " Model Performance",
    " Cost Predictor"
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🏥 Healthcare Cost & Utilisation Analyser")
    st.markdown("**CMS Medicare Inpatient Analysis — Provider Cost & Payment Intelligence Dashboard**")
    st.markdown("---")

    total_records    = len(df)
    total_discharges = df['total_discharges'].sum()
    avg_covered      = df['average_covered_charges'].mean()
    avg_payment      = df['average_total_payments'].mean()
    avg_medicare     = df['average_medicare_payments'].mean()
    states_covered   = df['provider_state'].nunique()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Records",       f"{total_records:,}")
    c2.metric("Total Discharges",    f"{total_discharges:,}")
    c3.metric("Avg Covered Charges", f"${avg_covered:,.0f}")
    c4.metric("Avg Total Payment",   f"${avg_payment:,.0f}")
    c5.metric("Avg Medicare Pay",    f"${avg_medicare:,.0f}")
    c6.metric("States Covered",      states_covered)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost Distribution by Category")
        cat_avg = df.groupby('drg_category')['average_total_payments'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors  = [CAT_COLORS.get(c, '#1565C0') for c in cat_avg.index]
        bars = ax.barh(cat_avg.index, cat_avg.values, color=colors)
        for bar, val in zip(bars, cat_avg.values):
            ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                    f'${val:,.0f}', va='center', fontsize=9)
        ax.set_xlabel("Avg Total Payment ($)")
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0, cat_avg.max() * 1.18)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Discharge Volume by Category")
        cat_vol = df.groupby('drg_category')['total_discharges'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors  = [CAT_COLORS.get(c, '#1565C0') for c in cat_vol.index]
        ax.bar(cat_vol.index, cat_vol.values, color=colors)
        ax.set_xticklabels(cat_vol.index, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("Total Discharges")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Regional Analysis":
    st.title("🗺️ Regional Cost Analysis")
    st.markdown("---")

    state_stats = df.groupby('provider_state').agg(
        avg_payment   =('average_total_payments',    'mean'),
        avg_covered   =('average_covered_charges',   'mean'),
        total_discharges=('total_discharges',        'sum'),
        record_count  =('provider_id',               'count')
    ).reset_index().sort_values('avg_payment', ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Highest Cost State", state_stats.iloc[0]['provider_state'],
                f"${state_stats.iloc[0]['avg_payment']:,.0f} avg")
    col2.metric("Lowest Cost State",  state_stats.iloc[-1]['provider_state'],
                f"${state_stats.iloc[-1]['avg_payment']:,.0f} avg")
    col3.metric("Cost Range",
                f"${state_stats['avg_payment'].max() - state_stats['avg_payment'].min():,.0f}",
                "between highest & lowest state")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 States by Avg Payment")
        top15 = state_stats.head(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        norm   = plt.Normalize(top15['avg_payment'].min(), top15['avg_payment'].max())
        colors = plt.cm.Blues(norm(top15['avg_payment']))
        ax.barh(top15['provider_state'][::-1], top15['avg_payment'][::-1], color=colors[::-1])
        ax.set_xlabel("Avg Total Payment ($)")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Covered Charges vs Actual Payment")
        sample = state_stats.head(15)
        x = np.arange(len(sample))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(x - 0.2, sample['avg_covered'],  0.35, label='Avg Covered Charges', color='#90CAF9')
        ax.bar(x + 0.2, sample['avg_payment'],  0.35, label='Avg Total Payment',   color='#1565C0')
        ax.set_xticks(x)
        ax.set_xticklabels(sample['provider_state'], rotation=45, fontsize=8)
        ax.set_ylabel("Amount ($)")
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Full State-Level Summary")
    display_df = state_stats.copy()
    display_df['avg_payment']   = display_df['avg_payment'].apply(lambda x: f"${x:,.0f}")
    display_df['avg_covered']   = display_df['avg_covered'].apply(lambda x: f"${x:,.0f}")
    display_df.columns = ['State','Avg Payment','Avg Covered','Total Discharges','Records']
    st.dataframe(display_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSIS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Diagnosis Insights":
    st.title("🔬 Diagnosis & Procedure Insights")
    st.markdown("---")

    selected_cat = st.selectbox("Filter by Category", ['All'] + sorted(df['drg_category'].unique()))
    filtered = df if selected_cat == 'All' else df[df['drg_category'] == selected_cat]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Costliest Procedures")
        top_drg = filtered.groupby('drg_definition')['average_total_payments'].mean()\
                          .sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors  = plt.cm.Reds_r(np.linspace(0.2, 0.8, len(top_drg)))
        ax.barh(
            [d[:35] + '...' if len(d) > 35 else d for d in top_drg.index[::-1]],
            top_drg.values[::-1], color=colors
        )
        ax.set_xlabel("Avg Total Payment ($)")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Top 10 by Discharge Volume")
        top_vol = filtered.groupby('drg_definition')['total_discharges'].sum()\
                          .sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors  = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(top_vol)))
        ax.barh(
            [d[:35] + '...' if len(d) > 35 else d for d in top_vol.index[::-1]],
            top_vol.values[::-1], color=colors
        )
        ax.set_xlabel("Total Discharges")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Cost-to-Charge Ratio by Category")
    st.caption("Lower ratio = Medicare pays a smaller fraction of what hospitals charge")
    ctc = df.groupby('drg_category')['cost_to_charge_ratio'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors  = [CAT_COLORS.get(c, '#1565C0') for c in ctc.index]
    ax.bar(ctc.index, ctc.values, color=colors)
    ax.set_xticklabels(ctc.index, rotation=20, ha='right')
    ax.set_ylabel("Cost-to-Charge Ratio")
    ax.axhline(ctc.mean(), color='red', linestyle='--', label=f'Avg: {ctc.mean():.2f}')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Covered Charges vs Payments — Scatter")
    fig, ax = plt.subplots(figsize=(8, 4))
    for cat, grp in df.groupby('drg_category'):
        ax.scatter(grp['average_covered_charges'], grp['average_total_payments'],
                   alpha=0.4, s=15, label=cat, color=CAT_COLORS.get(cat, '#1565C0'))
    ax.set_xlabel("Avg Covered Charges ($)")
    ax.set_ylabel("Avg Total Payment ($)")
    ax.legend(fontsize=7, ncol=2)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Predictive Model Performance")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest R²",  f"{results['rf_r2']:.4f}")
    c2.metric("Random Forest RMSE", f"${results['rf_rmse']:,.0f}")
    c3.metric("Random Forest MAE",  f"${results['rf_mae']:,.0f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Linear Reg. R²",   f"{results['lr_r2']:.4f}")
    c5.metric("Linear Reg. RMSE", f"${results['lr_rmse']:,.0f}")
    c6.metric("Linear Reg. MAE",  f"${results['lr_mae']:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted — Random Forest")
        y_test  = np.array(results['y_test'])
        rf_pred = np.array(results['rf_preds'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, rf_pred, alpha=0.3, s=10, color='#1565C0')
        mn, mx = min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Fit')
        ax.set_xlabel("Actual Payment ($)")
        ax.set_ylabel("Predicted Payment ($)")
        ax.legend(); ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Feature Importance — Random Forest")
        fi      = results['feature_importance']
        labels  = ['State', 'Category', 'DRG Code', 'Discharges', 'Covered Charges', 'Cost-Charge Ratio']
        values  = list(fi.values())
        sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], color='#1976D2')
        ax.set_xlabel("Feature Importance")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📌 Model Interpretation")
    st.success(f"""
**Random Forest dominates** with R² of {results['rf_r2']:.4f} and RMSE of ${results['rf_rmse']:,.0f} — meaning predictions are within ~${results['rf_rmse']:,.0f} of actual Medicare payments on average.

**Key cost drivers (by feature importance):**
- **Covered Charges** — the hospital's billed amount is the strongest predictor of actual payment
- **DRG Code** — the specific diagnosis/procedure determines the payment tier
- **Category** — procedure category (Orthopedic vs Respiratory) sets the baseline cost band
- **State** — geographic variation drives significant cost differences (NY/CA vs AL/OK)
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COST PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Cost Predictor":
    st.title("🔮 Medicare Payment Cost Predictor")
    st.markdown("Enter provider details to predict expected Medicare payment.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("Provider State", sorted(df['provider_state'].unique()))
        category = st.selectbox("Diagnosis Category", sorted(df['drg_category'].unique()))
        drg_options = sorted(df[df['drg_category'] == category]['drg_definition'].unique())
        drg = st.selectbox("DRG / Procedure", drg_options)

    with col2:
        discharges = st.slider("Total Discharges", 10, 500, 100)
        covered    = st.number_input("Avg Covered Charges ($)", min_value=1000,
                                      max_value=150000, value=20000, step=500)
        ctc_ratio  = st.slider("Cost-to-Charge Ratio", 0.20, 0.50, 0.33, step=0.01)

    if st.button("💰 Predict Medicare Payment", use_container_width=True):
        try:
            state_enc    = encoders['state'].transform([state])[0]
            category_enc = encoders['category'].transform([category])[0]
            drg_enc      = encoders['drg'].transform([drg])[0]

            input_data = pd.DataFrame([[state_enc, category_enc, drg_enc,
                                         discharges, covered, ctc_ratio]],
                                        columns=results['features'])

            rf_pred = rf.predict(input_data)[0]
            lr_pred = lr.predict(input_data)[0]
            ensemble = (rf_pred + lr_pred) / 2

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Random Forest Prediction", f"${rf_pred:,.0f}")
            c2.metric("Linear Reg. Prediction",   f"${lr_pred:,.0f}")
            c3.metric("Ensemble Estimate",         f"${ensemble:,.0f}")

            medicare_est = ensemble * 0.87
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
**💡 Cost Breakdown Estimate**
- Avg Covered Charges: **${covered:,.0f}**
- Predicted Total Payment: **${ensemble:,.0f}**
- Estimated Medicare Portion: **${medicare_est:,.0f}**
- Cost-to-Charge Ratio: **{ctc_ratio:.2f}**
                """)

            with col2:
                nat_avg = df['average_total_payments'].mean()
                delta   = ((ensemble - nat_avg) / nat_avg) * 100
                label   = "above" if delta > 0 else "below"
                st.warning(f"""
**📊 vs National Average**
- National Avg Payment: **${nat_avg:,.0f}**
- This Prediction: **${ensemble:,.0f}**
- **{abs(delta):.1f}% {label} national average**

{'⚠️ High-cost provider segment' if delta > 15 else '✅ Within normal cost range' if abs(delta) <= 15 else '🟢 Below-average cost provider'}
                """)

            # Gauge bar
            fig, ax = plt.subplots(figsize=(7, 1.2))
            max_val = df['average_total_payments'].max()
            ax.barh(0, max_val, color='#EEEEEE', height=0.4)
            color = '#E53935' if delta > 15 else '#43A047' if delta < -15 else '#1976D2'
            ax.barh(0, ensemble, color=color, height=0.4)
            ax.axvline(nat_avg, color='orange', linestyle='--', linewidth=1.5, label=f'National Avg ${nat_avg:,.0f}')
            ax.set_xlim(0, max_val)
            ax.set_yticks([])
            ax.set_xlabel("Predicted Payment ($)")
            ax.legend(fontsize=8)
            ax.spines[['top','right','left']].set_visible(False)
            st.pyplot(fig); plt.close()

        except Exception as e:
            st.error(f"Prediction error: {e}. Please check your inputs.")
