import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML for Heart Disease",
    page_icon="❤️",
    layout="wide"
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

.stApp {
    background: #0d0d0d;
    color: #f0f0f0;
}

.hero {
    position: relative;
    width: 100%;
    min-height: 280px;
    background: #000;
    border-radius: 0 0 24px 24px;
    overflow: hidden;
    display: flex;
    align-items: center;
    padding: 40px 60px;
    margin-bottom: 36px;
}
.hero-bg {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 75% 50%, rgba(180,0,30,0.35) 0%, transparent 70%),
        url("https://images.unsplash.com/photo-1530026405186-ed1ea0ac7a63?q=80&w=1400&auto=format&fit=crop")
        center/cover no-repeat;
    opacity: 0.55;
    z-index: 0;
}
.hero-ecg {
    position: absolute;
    bottom: 24px; left: 0; width: 100%; height: 60px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 60'%3E%3Cpolyline points='0,30 100,30 130,30 150,5 170,55 190,30 220,30 260,30 290,30 310,5 330,55 350,30 380,30 1200,30' stroke='%23ff1a3c' stroke-width='2' fill='none'/%3E%3C/svg%3E")
    repeat-x center/600px 60px;
    opacity: 0.6;
    z-index: 1;
}
.hero-content { position: relative; z-index: 2; }
.hero h1 {
    font-size: 2.6rem; font-weight: 800; color: #fff;
    letter-spacing: -0.5px; margin: 0 0 10px 0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.8);
}
.hero p {
    font-size: 1rem; color: #ccc; max-width: 500px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.8);
}

.section-title {
    font-size: 1.1rem; font-weight: 700;
    color: #ff3355; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 16px;
}

.metric-row { display:flex; gap:12px; margin-bottom:18px; flex-wrap:wrap; }
.metric-tile {
    flex:1; min-width:110px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 14px 16px; text-align:center;
}
.metric-val { font-size:1.5rem; font-weight:800; color:#ff3355; }
.metric-lbl { font-size:0.73rem; color:#aaa; margin-top:2px; }

.result-high {
    background: linear-gradient(135deg,rgba(200,0,30,0.25),rgba(120,0,20,0.35));
    border: 1.5px solid #cc0022;
    border-radius: 14px; padding: 22px 26px; margin-top: 16px;
}
.result-low {
    background: linear-gradient(135deg,rgba(0,170,90,0.2),rgba(0,100,50,0.3));
    border: 1.5px solid #00bb66;
    border-radius: 14px; padding: 22px 26px; margin-top: 16px;
}
.result-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 6px; }
.result-body  { font-size: 0.92rem; color: #ccc; }

div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
label { color: #ccc !important; font-size:0.85rem !important; }
div[data-baseweb="select"] > div { background:#1a1a1a !important; border-color:#333 !important; color:#fff !important; }
input[type="number"] { background:#1a1a1a !important; color:#fff !important; border-color:#333 !important; }
.stRadio label { color:#ccc !important; }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#cc0022,#ff3355) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight:700 !important;
    font-size:1rem !important; padding: 12px !important;
    letter-spacing:0.5px !important;
    box-shadow: 0 4px 18px rgba(200,0,30,0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD MODEL ARTIFACT (single .pkl file)
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    artifact  = joblib.load('heart_disease_logistic_model.pkl')
    model     = artifact['model']
    scaler    = artifact['scaler']
    features  = artifact['features']
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except Exception as e:
    st.error(f"⚠️ Could not load 'heart_disease_logistic_model.pkl': {e}")
    st.stop()

# ─────────────────────────────────────────────
# 4. HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-bg"></div>
  <div class="hero-ecg"></div>
  <div class="hero-content">
    <h1>❤️ Examining Your Hearts!</h1>
    <p>The goal of this project is to train a machine learning model to accurately predict
       whether a sample patient has been diagnosed with heart disease, with highest accuracy possible.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. METRIC TILES
# ─────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
  <div class="metric-tile"><div class="metric-val">87.0%</div><div class="metric-lbl">Test Accuracy</div></div>
  <div class="metric-tile"><div class="metric-val">93.3%</div><div class="metric-lbl">ROC-AUC Score</div></div>
  <div class="metric-tile"><div class="metric-val">88%</div><div class="metric-lbl">Precision (HD+)</div></div>
  <div class="metric-tile"><div class="metric-val">918</div><div class="metric-lbl">Training Samples</div></div>
  <div class="metric-tile"><div class="metric-val">15</div><div class="metric-lbl">Features Used</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1.15, 1], gap="large")

# ══════════════════════════════════════
# LEFT: PATIENT INPUTS
# ══════════════════════════════════════
with left_col:
    st.markdown("<div class='section-title'>📋 Heart's Data</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa;font-size:0.85rem;margin-bottom:14px;'>Input your data here</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: age          = st.number_input("Age", 1, 100, 50)
    with c2: sex          = st.selectbox("Sex", ["M", "F"])
    with c3: chest_pain   = st.selectbox("Chest Pain Type", ["ASY","ATA","NAP","TA"])
    with c4: resting_bp   = st.number_input("Resting BP", 50, 220, 130)

    c5, c6, c7, c8 = st.columns(4)
    with c5: cholesterol      = st.number_input("Cholesterol", 0, 600, 250)
    with c6: fasting_bs       = st.selectbox("FastingBS > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c7: resting_ecg      = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
    with c8: max_hr           = st.number_input("Max HR", 60, 220, 150)

    c9, c10, c11, _ = st.columns(4)
    with c9:  exercise_angina = st.selectbox("Ex. Angina", ["N","Y"])
    with c10: oldpeak         = st.number_input("Oldpeak", -3.0, 7.0, 0.0, step=0.1)
    with c11: st_slope        = st.selectbox("ST Slope", ["Up","Flat","Down"])

    st.write("")
    run_btn = st.button("🔥 ANALYSE", use_container_width=True)

# ══════════════════════════════════════
# RIGHT: CHART + RESULT
# ══════════════════════════════════════
with right_col:
    st.markdown("<div class='section-title'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Status": ["Heart Disease", "No Heart Disease"],
        "Count":  [508, 410]
    })
    fig = px.pie(
        chart_df, values="Count", names="Status",
        color_discrete_sequence=["#ff1a3c", "#2b2d42"],
        hole=0.5
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_traces(textfont_color='#fff')
    st.plotly_chart(fig, use_container_width=True)

    # ── PREDICTION ──
    if run_btn:

        # Step 1: Impute zeros (same as notebook cells 18 & 23)
        chol_val = cholesterol if cholesterol > 0 else 242.0   # training median
        bp_val   = resting_bp  if resting_bp  > 0 else 130.0   # training median

        # Step 2: IQR capping bounds (computed from training data, matches cell 27)
        CAPS = {
            'RestingBP':   (92.0,  162.0),
            'Cholesterol': (149.5, 360.5),
            'MaxHR':       (82.0,  195.0),
            'Oldpeak':     (-1.5,  4.5),
        }
        def cap(v, lo, hi): return float(max(lo, min(v, hi)))

        bp_val   = cap(bp_val,    *CAPS['RestingBP'])
        chol_val = cap(chol_val,  *CAPS['Cholesterol'])
        mhr_val  = cap(max_hr,    *CAPS['MaxHR'])
        op_val   = cap(oldpeak,   *CAPS['Oldpeak'])

        # Step 3: Build feature row matching get_dummies(drop_first=True) output
        row = {col: 0 for col in features}
        row['Age']         = age
        row['RestingBP']   = bp_val
        row['Cholesterol'] = chol_val
        row['FastingBS']   = fasting_bs
        row['MaxHR']       = mhr_val
        row['Oldpeak']     = op_val
        if sex == 'M':              row['Sex_M']             = 1
        if chest_pain == 'ATA':     row['ChestPainType_ATA'] = 1
        if chest_pain == 'NAP':     row['ChestPainType_NAP'] = 1
        if chest_pain == 'TA':      row['ChestPainType_TA']  = 1
        if resting_ecg == 'Normal': row['RestingECG_Normal'] = 1
        if resting_ecg == 'ST':     row['RestingECG_ST']     = 1
        if exercise_angina == 'Y':  row['ExerciseAngina_Y']  = 1
        if st_slope == 'Flat':      row['ST_Slope_Flat']     = 1
        if st_slope == 'Up':        row['ST_Slope_Up']       = 1

        # Step 4: Scale numerical columns only (same as cell 46)
        df_input = pd.DataFrame([row])[features]
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

        # Step 5: Predict
        pred  = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        # Risk gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={'suffix': '%', 'font': {'color': '#fff', 'size': 28}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#555'},
                'bar':  {'color': '#ff1a3c' if proba > 0.5 else '#00bb66'},
                'bgcolor': '#1a1a1a',
                'steps': [
                    {'range': [0,  40],  'color': 'rgba(0,187,102,0.15)'},
                    {'range': [40, 60],  'color': 'rgba(255,180,0,0.12)'},
                    {'range': [60, 100], 'color': 'rgba(255,26,60,0.15)'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.7, 'value': 50}
            },
            title={'text': 'Risk Score', 'font': {'color': '#aaa', 'size': 13}}
        ))
        gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=10),
            height=200,
            font={'color': '#ccc'}
        )
        st.plotly_chart(gauge, use_container_width=True)

        if pred == 1:
            st.markdown(f"""
            <div class="result-high">
              <div class="result-title">⚠️ High Risk Detected</div>
              <div class="result-body">
                The model flags a <strong style="color:#ff4466">{proba*100:.1f}%</strong> probability of cardiovascular complications.
                Please consult a healthcare professional for a thorough evaluation.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <div class="result-title">✅ Low Risk Profile</div>
              <div class="result-body">
                The model flags a <strong style="color:#00cc77">{(1-proba)*100:.1f}%</strong> probability of a normal cardiovascular profile.
                Continue maintaining a healthy lifestyle!
              </div>
            </div>""", unsafe_allow_html=True)
