from pathlib import Path

import io
import json
import numpy as np
import streamlit as st
import pandas as pd
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_prediction.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please make sure car_price_prediction.pkl is in the app folder.")
        return None

    return joblib.load(MODEL_PATH)


model = load_model()

if model is None:
    st.stop()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Initialize session state defaults for inputs
defaults = {
    "present_price": 5.0,
    "kms_driven": 10000,
    "owner": 0,
    "car_age": 3,
    "fuel": "Petrol",
    "seller": "Dealer",
    "transmission": "Manual",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(135deg, rgba(2, 6, 23, 0.86), rgba(15, 23, 42, 0.74), rgba(30, 64, 175, 0.66)),
            url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1800&q=80") center/cover fixed;
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #ffffff; /* default text color over background */
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2.2rem;
        max-width: 1500px;
    }
    /* Ensure page-level text on top of background is white */
    body, .stApp, .block-container, .stMarkdown, label, .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stCheckbox > label {
        color: #ffffff !important;
    }
    /* Keep text inside cards readable (dark) */
    .card, .hero-card, .info-pill, .card * {
        color: #0f172a !important;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #0f172a 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
    }
    .card {
        background: rgba(248, 250, 252, 0.97);
        padding: 1.25rem 1.35rem;
        border-radius: 22px;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18);
        border: 1px solid rgba(226, 232, 240, 0.96);
        backdrop-filter: blur(12px);
        height: 100%;
    }
    .hero-card {
        background: rgba(248, 250, 252, 0.96);
        border-radius: 26px;
        padding: 1.45rem 1.6rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(226, 232, 240, 0.96);
        backdrop-filter: blur(12px);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 1rem;
        align-items: start;
    }
    .hero-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 14px;
    }
    .badge {
        display: inline-block;
        padding: 0.38rem 0.8rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 0.8rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 2px 6px rgba(29, 78, 216, 0.15);
    }
    .info-pill {
        padding: 0.95rem 1rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd;
    }
    footer{margin-top:2rem; color:#94a3b8; font-size:0.9rem}
    /* Ensure hero area text is white against the backdrop */
    .hero-card, .hero-card * { color: #ffffff !important; }
    /* Keep card interior text dark for readability */
    .card, .card * { color: #020617 !important; }
    /* Info pills remain dark text for legibility */
    .info-pill { color: #020617 !important; }
    .badge { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
            <div style="display:flex; align-items:center; gap:0.9rem;">
                <div style="width:56px; height:56px; border-radius:10px; background:linear-gradient(135deg,#06b6d4,#059669); display:flex; align-items:center; justify-content:center; box-shadow:0 8px 24px rgba(6,182,212,0.08);">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 12h18" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                        <path d="M6 6h12" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size:0.85rem; color:rgba(255,255,255,0.8); font-weight:700; letter-spacing:0.08em;">AUTOBRAND</div>
                    <div style="font-size:1.6rem; font-weight:800; margin-top:2px;">Car Price Intelligence</div>
                </div>
            </div>
            <div style="display:flex; gap:0.6rem; align-items:center;">
                <button onclick="" style="background:transparent;border:1px solid rgba(255,255,255,0.06);color:rgba(255,255,255,0.9);padding:0.5rem 0.9rem;border-radius:10px;">Sign in</button>
                <button onclick="" style="background:linear-gradient(90deg,#06b6d4,#059669);border:none;color:#021024;padding:0.55rem 1rem;border-radius:10px;font-weight:700;">Get Estimate</button>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-top:1rem;">
            <img class="hero-image" src="https://images.unsplash.com/photo-1511919884226-fd3cad34687c?auto=format&fit=crop&w=900&q=80" alt="Luxury car" />
            <img class="hero-image" src="https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=900&q=80" alt="Sport car" />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1.2, 0.8, 0.8], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vehicle Details")
    st.caption("Fill in the key details below to estimate the resale value with greater clarity.")

    # Use two inner columns for inputs to use space well
    i1, i2 = st.columns([1, 1])

    with i1:
        present_price = st.number_input("💰 Present Price (Lakhs)", min_value=0.0, step=0.1, value=5.0, key="present_price")
        kms_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, step=1000, value=10000, key="kms_driven")
        owner = st.selectbox("👤 Previous Owners", [0, 1, 2, 3], key="owner")

    with i2:
        car_age = st.number_input("🧓 Car Age", min_value=0, step=1, value=3, key="car_age")
        fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel"], key="fuel")
        seller = st.selectbox("🧾 Seller Type", ["Dealer", "Individual"], key="seller")
        transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"], key="transmission")

    # Action buttons: Predict, Reset
    action_col1, action_col2, action_col3 = st.columns([1, 0.6, 0.8])
    with action_col1:
        predict_clicked = st.button("Predict Price", use_container_width=True)
    with action_col2:
        reset_clicked = st.button("Reset Inputs", use_container_width=True)
    with action_col3:
        save_clicked = st.button("Save Result", use_container_width=True)

    if reset_clicked:
        # reset session state keys to defaults
        st.session_state.present_price = 5.0
        st.session_state.kms_driven = 10000
        st.session_state.owner = 0
        st.session_state.car_age = 3
        st.session_state.fuel = "Petrol"
        st.session_state.seller = "Dealer"
        st.session_state.transmission = "Manual"
        st.experimental_rerun()

    result_json = None
    if predict_clicked:
        sample = pd.DataFrame(
            {
                "Present_Price": [present_price],
                "Kms_Driven": [kms_driven],
                "Owner": [owner],
                "Car_Age": [car_age],
                "Fuel_Type_Diesel": [1 if fuel == "Diesel" else 0],
                "Fuel_Type_Petrol": [1 if fuel == "Petrol" else 0],
                "Seller_Type_Individual": [1 if seller == "Individual" else 0],
                "Transmission_Manual": [1 if transmission == "Manual" else 0],
            }
        )

        prediction = model.predict(sample)
        predicted_price = float(prediction[0])

        # show result card
        st.markdown(
            f"""
            <div style="margin-top: 1rem; padding: 1.1rem 1.2rem; border-radius: 16px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #93c5fd;">
                <h3 style="margin:0 0 0.35rem 0; color:#1e3a8a; font-weight:900;">Estimated Selling Price</h3>
                <div style="font-size: 2rem; font-weight: 900; color:#020617;">₹ {predicted_price:.2f} Lakhs</div>
                <p style="margin:0.25rem 0 0 0; color:#0f172a; font-weight:700;">Based on the details you entered.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # prepare downloadable result
        result = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel": fuel,
            "Seller": seller,
            "Transmission": transmission,
            "Predicted_Price_Lakhs": predicted_price,
        }
        result_json = json.dumps(result, indent=2)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Why This Works")
    st.markdown(
        """
        <div style="padding:0.8rem 0.9rem; border-radius:14px; background:#f8fafc; border:1px solid #e2e8f0; color:#334155; line-height:1.6; font-weight:600;">
        This model considers the main factors that influence car resale value, including age, mileage, ownership, and fuel type.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Quick Tips")
    st.markdown("- Lower mileage often increases resale value")
    st.markdown("- Automatic cars may fetch a premium in some markets")
    st.markdown("- Well-maintained cars are usually valued higher")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Market Snapshot")
    st.metric("Typical Range", "₹ 3.5 - 12 Lakhs", "Updated")
    st.metric("Best Value", "Low Mileage", "Strong")
    st.metric("Main Driver", "Age & Kms", "High")
    st.markdown('</div>', unsafe_allow_html=True)

# Advanced: sensitivity charts and download
with st.expander("Advanced: Sensitivity & Export", expanded=False):
    st.markdown("### How price changes with Mileage and Age")
    # Only produce charts if model is available
    try:
        # Sensitivity for kilometers
        cur_kms = int(st.session_state.get("kms_driven", 10000))
        kmin = max(0, cur_kms - 40000)
        kmax = cur_kms + 40000
        kms_range = np.linspace(kmin, kmax, 50, dtype=int)

        cur_age = int(st.session_state.get("car_age", 3))
        ages = np.arange(0, max(8, cur_age + 5))

        sens_kms = []
        for k in kms_range:
            sf = pd.DataFrame({
                "Present_Price": [st.session_state.get("present_price", 5.0)],
                "Kms_Driven": [k],
                "Owner": [st.session_state.get("owner", 0)],
                "Car_Age": [st.session_state.get("car_age", 3)],
                "Fuel_Type_Diesel": [1 if st.session_state.get("fuel", "Petrol") == "Diesel" else 0],
                "Fuel_Type_Petrol": [1 if st.session_state.get("fuel", "Petrol") == "Petrol" else 0],
                "Seller_Type_Individual": [1 if st.session_state.get("seller", "Dealer") == "Individual" else 0],
                "Transmission_Manual": [1 if st.session_state.get("transmission", "Manual") == "Manual" else 0],
            })
            sens_kms.append(float(model.predict(sf)[0]))

        sens_age = []
        for a in ages:
            sf = pd.DataFrame({
                "Present_Price": [st.session_state.get("present_price", 5.0)],
                "Kms_Driven": [st.session_state.get("kms_driven", 10000)],
                "Owner": [st.session_state.get("owner", 0)],
                "Car_Age": [a],
                "Fuel_Type_Diesel": [1 if st.session_state.get("fuel", "Petrol") == "Diesel" else 0],
                "Fuel_Type_Petrol": [1 if st.session_state.get("fuel", "Petrol") == "Petrol" else 0],
                "Seller_Type_Individual": [1 if st.session_state.get("seller", "Dealer") == "Individual" else 0],
                "Transmission_Manual": [1 if st.session_state.get("transmission", "Manual") == "Manual" else 0],
            })
            sens_age.append(float(model.predict(sf)[0]))

        df_kms = pd.DataFrame({"Kms": kms_range, "Predicted_Lakhs": sens_kms}).set_index("Kms")
        df_age = pd.DataFrame({"Age": ages, "Predicted_Lakhs": sens_age}).set_index("Age")

        st.subheader("Mileage sensitivity")
        st.line_chart(df_kms)
        st.subheader("Age sensitivity")
        st.line_chart(df_age)

    except Exception as e:
        st.error(f"Could not build sensitivity charts: {e}")

    # Provide download if result available
    if 'result_json' in locals() and result_json is not None:
        st.download_button("Download result (JSON)", data=result_json, file_name="prediction_result.json", mime="application/json")
    else:
        st.info("Make a prediction first to download the result.")

st.markdown("<footer>Built with care • Contact: you@domain.com</footer>", unsafe_allow_html=True)