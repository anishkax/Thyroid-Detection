import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(page_title="🩺 Remi — Thyroid Health Detector", layout="wide")

# -------------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------------
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    model = None
    scaler = None

label_map = {0: "Negative", 1: "Hypothyroid", 2: "Hyperthyroid"}

# -------------------------------------------------------
# LOAD CSS
# -------------------------------------------------------
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ styles.css not found — skipping custom styling.")

# -------------------------------------------------------
# NAVIGATION BAR
# -------------------------------------------------------
pages = ["Home", "Single Prediction", "Batch Prediction", "About", "Contact"]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: white;
        color: #015A84;
        padding: 1rem 0;
        border-bottom: 2px solid #015A84;
        margin-bottom: 1rem;
        gap: 1.5rem;
    }
    .stButton>button {
        background-color: transparent;
        color: #015A84;
        border: none;
        font-weight: 600;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        color: #FFB703;
        transform: scale(1.05);
    }
    .active-btn {
        color: #FFB703 !important;
        border-bottom: 2px solid #FFB703;
        padding-bottom: 3px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='navbar'>", unsafe_allow_html=True)
cols = st.columns(len(pages))
for i, p in enumerate(pages):
    active_class = "active-btn" if st.session_state["page"] == p else ""
    with cols[i]:
        if st.button(p, key=p):
            st.session_state["page"] = p
            st.rerun()
        st.markdown(f"<p class='{active_class}' style='text-align:center;'>{'•' if active_class else ''}</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# HELPER FUNCTION — LOAD HTML PROPERLY
# -------------------------------------------------------
def load_html(path):
    """Render HTML in Streamlit"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read().strip()
        html = html.replace("```html", "").replace("```", "")
        st.components.v1.html(html, height=700, scrolling=True)
    except FileNotFoundError:
        st.error(f"⚠️ Could not find file: {path}")
    except Exception as e:
        st.error(f"⚠️ Error loading {path}: {e}")

# -------------------------------------------------------
# PAGE LOGIC
# -------------------------------------------------------
page = st.session_state["page"]

if page == "Home":
    load_html("Components/Home.html")

elif page == "Single Prediction":
    st.header("👤 Predict Thyroid Status for One Patient")
    st.markdown("Please fill in the details below based on your medical reports.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (in years)", min_value=0, max_value=120)
        TSH = st.number_input("TSH (Thyroid Stimulating Hormone)", min_value=0.0, max_value=500.0)
        T3 = st.number_input("T3 (Triiodothyronine)", min_value=0.0, max_value=10.0)
        TT4 = st.number_input("TT4 (Total Thyroxine)", min_value=0.0, max_value=500.0)
        T4U = st.number_input("T4U (Thyroxine Uptake)", min_value=0.0, max_value=10.0)
        FTI = st.number_input("FTI (Free Thyroxine Index)", min_value=0.0, max_value=500.0)

    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex == "Male" else 0

        st.markdown("### 🧬 Medical & Clinical History (Select Yes / No)")
        def yn(label): return st.radio(label, ["No", "Yes"], horizontal=True)

        on_thyroxine = 1 if yn("Currently on Thyroxine medication") == "Yes" else 0
        query_on_thyroxine = 1 if yn("Doctor suspects Thyroxine need") == "Yes" else 0
        on_antithyroid_meds = 1 if yn("Currently taking Anti-thyroid medication") == "Yes" else 0
        sick = 1 if yn("Recently Ill") == "Yes" else 0
        pregnant = 1 if yn("Currently Pregnant") == "Yes" else 0
        thyroid_surgery = 1 if yn("Had Thyroid Surgery") == "Yes" else 0
        I131_treatment = 1 if yn("Had I131 Radioactive Treatment") == "Yes" else 0
        query_hypothyroid = 1 if yn("Doctor suspects Hypothyroidism") == "Yes" else 0
        query_hyperthyroid = 1 if yn("Doctor suspects Hyperthyroidism") == "Yes" else 0
        lithium = 1 if yn("Taking Lithium medication") == "Yes" else 0
        goitre = 1 if yn("Has Goitre (Neck Swelling)") == "Yes" else 0
        tumor = 1 if yn("Has Thyroid Tumor") == "Yes" else 0
        hypopituitary = 1 if yn("Pituitary gland issue") == "Yes" else 0
        psych = 1 if yn("Psychiatric history") == "Yes" else 0

    input_data = {
        "age": age, "sex": sex, "on_thyroxine": on_thyroxine, "query_on_thyroxine": query_on_thyroxine,
        "on_antithyroid_meds": on_antithyroid_meds, "sick": sick, "pregnant": pregnant,
        "thyroid_surgery": thyroid_surgery, "I131_treatment": I131_treatment,
        "query_hypothyroid": query_hypothyroid, "query_hyperthyroid": query_hyperthyroid,
        "lithium": lithium, "goitre": goitre, "tumor": tumor, "hypopituitary": hypopituitary,
        "psych": psych, "TSH": TSH, "T3": T3, "TT4": TT4, "T4U": T4U, "FTI": FTI
    }

    input_df = pd.DataFrame([input_data])

    if st.button("🔍 Predict Result"):
        if model is None:
            st.error("⚠️ Model not found. Please check model.pkl file.")
        else:
            try:
                if scaler is not None:
                    input_df[input_df.columns] = scaler.transform(input_df)
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
                confidence = round(max(prob) * 100, 2)
                st.success(f"### 🧾 Prediction: {label_map[pred]}")
                st.info(f"Confidence Level: **{confidence}%**")

                prob_df = pd.DataFrame({
                    "Condition": [label_map[i] for i in range(len(prob))],
                    "Probability (%)": [round(p * 100, 2) for p in prob]
                })
                st.table(prob_df)
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")

elif page == "Batch Prediction":
    load_html("Components/Batch_prediction.html")

elif page == "About":
    load_html("Components/About.html")

elif page == "Contact":
    load_html("Components/Contact.html")
