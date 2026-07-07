from pathlib import Path

import io
import json
import warnings
import logging
from typing import Optional, Dict

import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Reduce noisy Streamlit "missing ScriptRunContext" logs
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Suppress sklearn version mismatch warning when unpickling models saved with
# an older scikit-learn. This prevents the long InconsistentVersionWarning
# messages from cluttering the logs while keeping functionality.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    # If sklearn isn't installed yet in this environment, just continue;
    # installation should be performed in the project's venv.
    pass

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_prediction.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please make sure car_price_prediction.pkl is in the app folder.")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

model = load_model()

if model is None:
    st.stop()


def prepare_sample_df(present_price, kms_driven, owner, car_age, fuel, seller, transmission, model):
    """Build a DataFrame with column names matching the trained model's
    expected features. Handles both raw categorical features (e.g. 'Fuel_Type')
    and one-hot encoded features (e.g. 'Fuel_Type_Petrol').
    """
    base = {
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Car_Age": car_age,
    }

    # Try to read feature names the model was trained with
    feature_names = getattr(model, "feature_names_in_", None)

    # Helper to fill common categorical representations
    def fill_categorical(d):
        # Provide both string and numeric encodings to maximize compatibility
        # Numeric mappings used in the notebook training pipeline
        fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
        seller_map = {"Dealer": 0, "Individual": 1}
        trans_map = {"Manual": 0, "Automatic": 1}

        d.setdefault("Fuel_Type", fuel_map.get(fuel, fuel))
        d.setdefault("Seller_Type", seller_map.get(seller, seller))
        d.setdefault("Transmission", trans_map.get(transmission, transmission))
        d.setdefault("Fuel_Type_Diesel", 1 if fuel == "Diesel" else 0)
        d.setdefault("Fuel_Type_Petrol", 1 if fuel == "Petrol" else 0)
        d.setdefault("Seller_Type_Individual", 1 if seller == "Individual" else 0)
        d.setdefault("Transmission_Manual", 1 if transmission == "Manual" else 0)

    if feature_names is None:
        # Unknown feature names — include both raw and one-hot forms to maximize
        # compatibility. The model will ignore unknown columns automatically.
        fill_categorical(base)
        return pd.DataFrame([base])

    # Build row dict matching the model feature order
    row = {}

    # Provide mapping values for raw categorical features
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 0, "Automatic": 1}

    for fname in feature_names:
        if fname in base:
            row[fname] = base[fname]
        elif fname == "Fuel_Type":
            # encode fuel as numeric if model expects numeric
            row[fname] = fuel_map.get(fuel, fuel)
        elif fname == "Seller_Type":
            row[fname] = seller_map.get(seller, seller)
        elif fname == "Transmission":
            row[fname] = trans_map.get(transmission, transmission)
        elif fname == "Fuel_Type_Diesel":
            row[fname] = 1 if fuel == "Diesel" else 0
        elif fname == "Fuel_Type_Petrol":
            row[fname] = 1 if fuel == "Petrol" else 0
        elif fname == "Seller_Type_Individual":
            row[fname] = 1 if seller == "Individual" else 0
        elif fname == "Transmission_Manual":
            row[fname] = 1 if transmission == "Manual" else 0
        else:
            # Unknown feature expected by model — fill a safe default
            row[fname] = 0

    return pd.DataFrame([row])


@st.cache_data(ttl=300)
def predict_price(sample_df: pd.DataFrame) -> Optional[float]:
    """Predict price using the loaded model. Results are cached for 5 minutes.

    Returns predicted price as float or None on failure.
    """
    try:
        pred = model.predict(sample_df)
        return float(pred[0])
    except Exception as e:
        logger.exception("Prediction error")
        return None


def validate_inputs(present_price: float, kms_driven: int, owner: int, car_age: int) -> Optional[str]:
    """Return None if inputs valid, otherwise an error message string."""
    if present_price < 0:
        return "Present price must be non-negative."
    if kms_driven < 0:
        return "Kilometers driven must be non-negative."
    if owner < 0:
        return "Owner count must be non-negative."
    if car_age < 0:
        return "Car age must be non-negative."
    return None


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

if "result_json" not in st.session_state:
    st.session_state["result_json"] = None

if "predicted_price" not in st.session_state:
    st.session_state["predicted_price"] = None

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

    # --- Helper to reset inputs safely via button callback ---
    def _reset_inputs():
        st.session_state.update(
            {
                "present_price": 5.0,
                "kms_driven": 10000,
                "owner": 0,
                "car_age": 3,
                "fuel": "Petrol",
                "seller": "Dealer",
                "transmission": "Manual",
            }
        )
            # No explicit rerun call: Streamlit will re-run the script after the
            # callback completes. Calling `st.experimental_rerun()` is not
            # available in all Streamlit builds (some environments raise
            # AttributeError). Avoid calling it directly to preserve
            # compatibility; updating `st.session_state` is sufficient.

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
        # Use on_click to update session_state before widgets render
        reset_clicked = st.button("Reset Inputs", on_click=_reset_inputs, use_container_width=True)
    with action_col3:
        save_clicked = st.button("Save Result", use_container_width=True)

    # Note: reset is handled by the on_click callback `_reset_inputs` above.

    if predict_clicked:
        err = validate_inputs(present_price, kms_driven, owner, car_age)
        if err is not None:
            st.error(err)
            st.session_state.predicted_price = None
        else:
            sample = prepare_sample_df(
                present_price,
                kms_driven,
                owner,
                car_age,
                fuel,
                seller,
                transmission,
                model,
            )
            st.session_state.predicted_price = predict_price(sample)
            if st.session_state.predicted_price is None:
                st.error("Prediction failed; check model and inputs.")

        if st.session_state.predicted_price is not None:
            st.markdown(
                f"""
                <div style="margin-top: 1rem; padding: 1.1rem 1.2rem; border-radius: 16px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #93c5fd;">
                    <h3 style="margin:0 0 0.35rem 0; color:#1e3a8a; font-weight:900;">Estimated Selling Price</h3>
                    <div style="font-size: 2rem; font-weight: 900; color:#020617;">₹ {st.session_state.predicted_price:,.2f} Lakhs</div>
                    <p style="margin:0.25rem 0 0 0; color:#0f172a; font-weight:700;">Based on the details you entered.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        result = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel": fuel,
            "Seller": seller,
            "Transmission": transmission,
            "Predicted_Price_Lakhs": st.session_state.predicted_price,
        }
        st.session_state.result_json = json.dumps(result, indent=2)

    if save_clicked:
        if st.session_state.result_json is not None:
            st.download_button(
                "Download result (JSON)",
                data=st.session_state.result_json,
                file_name="prediction_result.json",
                mime="application/json",
            )
        else:
            st.info("Make a prediction first to save the result.")

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
            sf = prepare_sample_df(
                st.session_state.get("present_price", 5.0),
                int(k),
                st.session_state.get("owner", 0),
                st.session_state.get("car_age", 3),
                st.session_state.get("fuel", "Petrol"),
                st.session_state.get("seller", "Dealer"),
                st.session_state.get("transmission", "Manual"),
                model,
            )
            p = predict_price(sf)
            sens_kms.append(float(p) if p is not None else float('nan'))

        sens_age = []
        for a in ages:
            sf = prepare_sample_df(
                st.session_state.get("present_price", 5.0),
                st.session_state.get("kms_driven", 10000),
                st.session_state.get("owner", 0),
                int(a),
                st.session_state.get("fuel", "Petrol"),
                st.session_state.get("seller", "Dealer"),
                st.session_state.get("transmission", "Manual"),
                model,
            )
            p = predict_price(sf)
            sens_age.append(float(p) if p is not None else float('nan'))

        df_kms = pd.DataFrame({"Kms": kms_range, "Predicted_Lakhs": sens_kms}).set_index("Kms")
        df_age = pd.DataFrame({"Age": ages, "Predicted_Lakhs": sens_age}).set_index("Age")

        st.subheader("Mileage sensitivity")
        st.line_chart(df_kms)
        st.subheader("Age sensitivity")
        st.line_chart(df_age)

    except Exception as e:
        st.error(f"Could not build sensitivity charts: {e}")

    # Provide download if result available in session state
    if st.session_state.result_json is not None:
        st.download_button(
            "Download result (JSON)",
            data=st.session_state.result_json,
            file_name="prediction_result.json",
            mime="application/json",
        )
    else:
        st.info("Make a prediction first to download the result.")

st.markdown("<footer>Built with care • Contact: you@domain.com</footer>", unsafe_allow_html=True)