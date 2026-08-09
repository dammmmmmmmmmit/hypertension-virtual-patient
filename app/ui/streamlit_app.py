"""
Streamlit UI: a free-text box for the compound(s)/question (per Section
10), plus a structured patient-profile form — deliberately NOT parsed
from prose, since numeric vitals (BP, egfr, potassium) are more reliably
entered directly than guessed/defaulted by an LLM. See
app/agent/parse_query.py's patient-already-supplied branch.

Run: uv run streamlit run app/ui/streamlit_app.py
Requires a real ANTHROPIC_API_KEY in .env — the LLM nodes (parse_query,
generate_report) are not yet live-tested in this session, see DECISIONS.md.
"""

import asyncio

import streamlit as st

from app.agent.graph import run_agent
from app.core.drug_registry import HYPERTENSION_DRUGS
from app.schemas.patient import Comorbidity, DiseaseParameters, PatientProfile, Sex

st.set_page_config(page_title="Virtual Patient Drug-Response Simulator", layout="wide")

st.title("Virtual Patient Drug-Response Simulator")
st.caption(
    "Hypertension-only screening tool. Not clinical guidance. "
    "See README.md / DECISIONS.md for scope and limitations."
)

registry_names = sorted(d["name"] for d in HYPERTENSION_DRUGS)

with st.sidebar:
    st.header("Patient profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=58)
    sex = st.selectbox("Sex", options=[Sex.MALE, Sex.FEMALE], format_func=lambda s: s.value)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=82.0)

    st.subheader("Baseline vitals")
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=250.0, value=152.0)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=96.0)
    heart_rate = st.number_input("Heart rate (bpm)", min_value=30.0, max_value=200.0, value=78.0)
    egfr = st.number_input("eGFR (renal function, mL/min/1.73m2)", min_value=5.0, max_value=140.0, value=90.0)
    serum_potassium = st.number_input("Serum potassium (mmol/L)", min_value=2.5, max_value=7.0, value=4.2)

    st.subheader("Comorbidities")
    comorbidities = st.multiselect(
        "Select any that apply",
        options=[c for c in Comorbidity if c != Comorbidity.NONE],
        format_func=lambda c: c.value.replace("_", " "),
    )

    with st.expander("Registered drugs in this tool"):
        st.write(", ".join(registry_names))

st.subheader("Describe the compound(s) and your question")
raw_query = st.text_area(
    "e.g. \"What's the predicted effect of combining amlodipine and losartan?\"",
    height=100,
)

run_clicked = st.button("Run simulation", type="primary")

if run_clicked:
    if not raw_query.strip():
        st.warning("Enter a query describing the compound(s) you want to simulate.")
    else:
        patient = PatientProfile(
            age=age,
            sex=sex,
            weight_kg=weight_kg,
            baseline=DiseaseParameters(
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                egfr=egfr,
                serum_potassium=serum_potassium,
            ),
            comorbidities=comorbidities or [Comorbidity.NONE],
        )

        with st.spinner("Running simulation..."):
            try:
                final_state = asyncio.run(run_agent(raw_query, patient=patient))
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                final_state = None

        if final_state:
            if final_state.get("discouraged_warning"):
                st.error(f"⚠️ {final_state['discouraged_warning']}")

            st.markdown("### Report")
            st.markdown(final_state.get("report", "(no report generated)"))

            prediction = final_state.get("prediction")
            if prediction:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted systolic Δ", f"{prediction['deltas'][0].predicted_delta:+.1f} mmHg")
                with col2:
                    st.metric("Predicted diastolic Δ", f"{prediction['deltas'][1].predicted_delta:+.1f} mmHg")
                st.caption(f"Confidence (bioactivity-evidence heuristic, not calibrated): {prediction['confidence']}")

                with st.expander("Top predicted side effects"):
                    for name, prob in prediction["side_effect_probabilities"].items():
                        st.write(f"{name}: {prob:.2f}")

            with st.expander("Limitations / caveats (always shown)"):
                for limitation in final_state.get("limitations", []):
                    st.write(f"- {limitation}")
