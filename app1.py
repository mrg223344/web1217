import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import io
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# -------------------------- Page Configuration --------------------------
st.set_page_config(
    page_title="RF ML-Based Malignancies Prediction in CVD Patients",
    page_icon="🧬",
    layout="centered"
)

# -------------------------- Embedded Dataset (Fallback Training) --------------------------
RAW_DATA = """FAST	total_chol	uric_acid	ALP	age	HbA1c	ALT	BMI	lbxsassi	outcome
0.079806938	161	5.7	72	47	5.3	32	24.6	22	0
0.011809511	147	4.7	105	80	5.6	18	25.1	15	0
0.002318498	185	4.1	103	44	10	17	38.2	10	0
0.029564564	169	6.6	73	76	6.9	20	21.5	18	0
0.215115968	179	8.7	152	62	5.9	77	36.4	26	1
0.725956624	86	4.7	81	62	4.1	42	41.7	36	0
0.272158929	186	7.4	54	80	5.1	36	25.5	31	0
0.191901016	146	3.7	89	80	5.9	26	27.6	33	1
0.516882024	226	7.2	64	30	13.5	57	61.9	29	0
0.008087394	191	7.3	99	65	5.4	26	23.1	12	0
0.486336778	271	8.1	63	74	7.7	37	34	31	0
0.104534429	184	7.2	76	80	5.7	58	29	24	1
0.087201742	219	5.2	64	56	5.4	14	35.7	20	0
0.024853707	123	7.3	85	80	5.6	15	31.6	15	0
0.36903082	158	9	73	80	9.3	25	34.2	45	1
0.091645486	140	5.7	77	52	6	63	42.1	15	0
0.030092601	133	7	68	80	7.5	23	27.6	13	0
0.053388561	162	5.2	124	47	5.7	49	35.2	15	0
0.055277211	166	8.8	65	78	6.9	18	26.5	19	0
0.088202468	157	4.2	98	80	5.8	30	29.5	16	0
0.083421301	142	5.3	96	66	5.6	24	33.7	13	1
0.018428841	177	9.5	103	80	5.7	115	28.5	14	0
0.004839559	197	6.8	109	66	10.3	18	45.3	10	0
0.033297624	186	5.3	66	67	5.5	12	19	18	0
0.067696518	180	6	71	55	5.2	79	26.9	18	0
0.13486175	175	5.7	80	80	5.8	12	20.6	28	0
0.302889108	170	9.5	62	80	5.1	48	28.2	25	0
0.365483826	181	5.2	85	80	5.5	84	19.7	43	1
0.053604104	222	4.3	84	57	5.7	68	23.1	21	0
0.016576093	217	8.2	109	80	5.9	12	39.4	14	0
0.224325319	150	5.4	85	59	8.1	12	26.4	24	0
0.194960029	187	6.6	96	58	5.7	36	28.1	39	0"""

@st.cache_resource
def get_model_and_explainer():
    model_path = "rf_model.pkl"

    if not os.path.exists(model_path):
        df = pd.read_csv(io.StringIO(RAW_DATA), sep='\t')
        X = df.drop('outcome', axis=1)
        y = df['outcome']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)
    return model, explainer

rf_model, shap_explainer = get_model_and_explainer()

# -------------------------- Main Title --------------------------
st.title("🧠 RF Machine Learning–Based Malignancies Prediction Model")
st.markdown(
    "### Risk Prediction and SHAP-Based Interpretation in Cardiovascular Disease (CVD) Patients"
)
st.divider()

# -------------------------- Feature Input --------------------------
col1, col2, col3 = st.columns(3)

with col1:
    FAST = st.number_input("FAST", 0.0, 1.0, 0.0118, step=0.001, format="%.6f")
    total_chol = st.number_input("Total Cholesterol", 0, 400, 147)
    uric_acid = st.number_input("Uric Acid", 0.0, 20.0, 4.7, step=0.1)

with col2:
    ALP = st.number_input("ALP", 0, 400, 105)
    age = st.number_input("Age", 0, 120, 80)
    HbA1c = st.number_input("HbA1c (%)", 0.0, 20.0, 5.6, step=0.1)

with col3:
    ALT = st.number_input("ALT", 0, 300, 18)
    BMI = st.number_input("BMI (kg/m²)", 0.0, 60.0, 25.1, step=0.1)
    AST_display = st.number_input("AST", 0, 100, 15)  # ✅ only display name changed

st.divider()

# -------------------------- Prediction & SHAP --------------------------
input_data = pd.DataFrame({
    "FAST": [FAST],
    "total_chol": [total_chol],
    "uric_acid": [uric_acid],
    "ALP": [ALP],
    "age": [age],
    "HbA1c": [HbA1c],
    "ALT": [ALT],
    "BMI": [BMI],
    "lbxsassi": [AST_display]  # ✅ keep backend variable name unchanged
})

if st.button("🚀 Run Prediction and SHAP Analysis", type="primary"):
    with st.spinner("Running model inference and SHAP explanation..."):
        pred_label = rf_model.predict(input_data)[0]
        pred_prob = rf_model.predict_proba(input_data)[0]

        target_class = 1
        shap_values = shap_explainer(input_data)
        shap_exp = shap_values[0, :, target_class]

        st.success("✅ Prediction Completed")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Outcome", int(pred_label), help="0 = Non-malignant, 1 = Malignant")
        with col_b:
            st.metric("Probability of Malignancy", f"{pred_prob[1]:.4f}")

        st.divider()

        st.subheader("🔍 SHAP Waterfall Plot (Individual Explanation)")
        st.markdown(f"""
        This plot illustrates how each clinical feature contributes to the final prediction.
        - **Red bars**: Increased malignancy risk  
        - **Blue bars**: Decreased malignancy risk  

        **Baseline (E[f(x)])**: {shap_exp.base_values:.4f}  
        **Model Output (f(x))**: {(shap_exp.values.sum() + shap_exp.base_values):.4f}
        """)

        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_exp, show=False, max_display=10)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("View Input Data"):
            st.dataframe(input_data)

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.header("ℹ️ About This Model")
    st.info("""
    This application implements a **Random Forest machine learning model**
    for malignancy risk prediction in patients with cardiovascular disease (CVD).

    **SHAP (SHapley Additive exPlanations)** is used to provide transparent,
    individualized interpretation of model predictions.
    """)
