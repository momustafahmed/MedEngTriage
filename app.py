import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import json

# ---------------- Basic setup ----------------
st.set_page_config(page_title="Medical Triage System", layout="centered")

# Subtle top spacing
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# Load fitted pipeline and (optional) label encoder
pipe = load("models/best_pipe.joblib")
try:
    le = load("models/label_encoder.joblib")
except Exception:
    le = None

# Load feature schema if available (for correct column order/types)
CAT_FALLBACK = [
    "Has_Fever","Fever_Level","Fever_Duration_Level","Chills",
    "Has_Cough","Cough_Type","Cough_Duration_Level","Blood_Cough","Breath_Difficulty",
    "Has_Headache","Headache_Severity","Headache_Duration_Level","Photophobia","Neck_Stiffness",
    "Has_Abdominal_Pain","Pain_Location","Pain_Duration_Level","Nausea","Diarrhea",
    "Has_Fatigue","Fatigue_Severity","Fatigue_Duration_Level","Weight_Loss","Fever_With_Fatigue",
    "Has_Vomiting","Vomiting_Severity","Vomiting_Duration_Level","Blood_Vomit","Unable_To_Keep_Fluids",
    "Age_Group"
]
NUM_FALLBACK = ["Red_Flag_Count"]

try:
    with open("ui_assets/feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    CAT_COLS = schema.get("cat_cols", CAT_FALLBACK)
    NUM_COLS = schema.get("num_cols", NUM_FALLBACK)
except Exception:
    CAT_COLS, NUM_COLS = CAT_FALLBACK, NUM_FALLBACK

EXPECTED_COLS = CAT_COLS + NUM_COLS

# --------------- Choices (English) ---------------
YN = ["Yes", "No"]
SEV = ["Mild", "Moderate", "Severe"]
COUGH_TYPE = ["Dry", "Wet"]
PAIN_LOC = ["Upper abdomen", "Lower abdomen", "Entire abdomen"]
AGE_GROUP = ["Child", "Adult", "Elderly"]

# Map UI selections (English) back to model tokens (Somali)
YN_TO_TOKEN = {"Yes": "haa", "No": "maya"}
SEV_TO_TOKEN = {"Mild": "fudud", "Moderate": "dhexdhexaad", "Severe": "aad u daran"}
COUGH_TO_TOKEN = {"Dry": "qalalan", "Wet": "qoyan"}
PAINLOC_TO_TOKEN = {
    "Upper abdomen": "caloosha sare",
    "Lower abdomen": "caloosha hoose",
    "Entire abdomen": "caloosha oo dhan",
}

# Duration mapping: show phrases, map to model tokens
DUR_TOKEN_TO_DISPLAY = {
    "fudud": "Less than 1 day",
    "dhexdhexaad": "2-3 days",
    "dhexdhexaad ah": "2-3 days",
    "aad u daran": "More than 3 days",
}
# When user picks a phrase, convert back to token for model input
DUR_DISPLAY_TO_TOKEN = {
    v: ("dhexdhexaad" if k.startswith("dhexdhexaad") else k)
    for k, v in DUR_TOKEN_TO_DISPLAY.items()
}
DUR_DISPLAY = list(dict.fromkeys(DUR_TOKEN_TO_DISPLAY.values()))

# --------------- Default one-sentence tips ---------------
TRIAGE_TIPS = {
    "Mild condition (Home care)":
        "Rest at home, drink plenty of fluids, eat light meals, take pain relievers or fever reducers if needed, monitor your symptoms for 24 hours, if they worsen contact a healthcare facility.",
    "Moderate condition (Outpatient care)":
        "Visit a healthcare facility within 24 hours for evaluation, bring any previous medication records if available, drink plenty of fluids.",
    "Moderate condition (Outpatient care)":
        "Visit a healthcare facility within 24 hours for evaluation, bring any previous medication records if available, drink plenty of fluids.",
    "Emergency condition":
        "Go to the hospital immediately, do not attempt home treatment, if possible have someone accompany you, bring any previous medication records if available."
}
## Map model output labels (Somali) to English for UI display
SOMALI_TO_ENGLISH_LABEL = {
    "Xaalad fudud (Daryeel guri)": "Mild condition (Home care)",
    "Xaalad dhax dhaxaad eh (Bukaan socod)": "Moderate condition (Outpatient care)",
    "Xaalad dhax dhaxaad ah (Bukaan socod)": "Moderate condition (Outpatient care)",
    "Xaalad deg deg ah": "Emergency condition",
}
EXTRA_NOTICE = (
    "Important notice: This is a general assessment to help you understand your condition and next steps. "
    "If you are concerned about your condition, contact a healthcare provider."
)

# --------------- Helpers ---------------
def make_input_df(payload: dict) -> pd.DataFrame:
    """Ensure types are model-friendly (avoid isnan/type errors)."""
    row = {c: np.nan for c in EXPECTED_COLS}
    row.update(payload or {})

    # Categorical as object, numeric coerced
    for c in CAT_COLS:
        v = row.get(c, np.nan)
        if v is None:
            row[c] = np.nan
        else:
            s = str(v).strip()
            row[c] = np.nan if s == "" else s

    for c in NUM_COLS:
        try:
            row[c] = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        except Exception:
            row[c] = np.nan

    df_one = pd.DataFrame([row])
    for c in CAT_COLS:
        df_one[c] = df_one[c].astype("object")
    return df_one

def decode_label(y):
    """Return Somali label from model output."""
    try:
        if le is not None and isinstance(y, (int, np.integer)):
            return le.inverse_transform([y])[0]
    except Exception:
        pass
    return str(y)

def triage_style(label_so: str):
    """
    Return (bg, text, border) for a light, readable card.
    Green (home care), Amber (outpatient), Red (emergency).
    """
    t = (label_so or "").lower()
    if "emergency" in t or "urgent" in t:
        return ("#FFEBEE", "#B71C1C", "#EF9A9A")
    if "moderate" in t or "outpatient" in t:
        return ("#FFF8E1", "#8D6E00", "#FFD54F")
    return ("#E8F5E9", "#1B5E20", "#A5D6A7")

def render_select(label, wtype, key):
    placeholder = "Select"
    if wtype == "yn":
        val = st.selectbox(label, YN, index=None, placeholder=placeholder, key=key)
        return None if val is None else YN_TO_TOKEN.get(val, val)
    if wtype == "sev":
        val = st.selectbox(label, SEV, index=None, placeholder=placeholder, key=key)
        return None if val is None else SEV_TO_TOKEN.get(val, val)
    if wtype == "cough":
        val = st.selectbox(label, COUGH_TYPE, index=None, placeholder=placeholder, key=key)
        return None if val is None else COUGH_TO_TOKEN.get(val, val)
    if wtype == "painloc":
        val = st.selectbox(label, PAIN_LOC, index=None, placeholder=placeholder, key=key)
        return None if val is None else PAINLOC_TO_TOKEN.get(val, val)
    if wtype == "dur":
        disp = st.selectbox(label, DUR_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        # DUR_DISPLAY_TO_TOKEN already returns Somali token
        return DUR_DISPLAY_TO_TOKEN.get(disp, disp)
    return None

# --------------- Symptom groups (English) ---------------
SYMPTOMS = {
    "Fever": {
        "flag": "Has_Fever",
        "fields": [
            ("Fever_Level", "Fever severity", "sev"),
            ("Fever_Duration_Level", "Fever duration", "dur"),
            ("Chills", "Chills", "yn"),
        ],
    },
    "Cough": {
        "flag": "Has_Cough",
        "fields": [
            ("Cough_Type", "Type of cough", "cough"),
            ("Cough_Duration_Level", "Cough duration", "dur"),
            ("Blood_Cough", "Blood in cough", "yn"),
            ("Breath_Difficulty", "Breathing difficulty", "yn"),
        ],
    },
    "Headache": {
        "flag": "Has_Headache",
        "fields": [
            ("Headache_Severity", "Headache severity", "sev"),
            ("Headache_Duration_Level", "Headache duration", "dur"),
            ("Photophobia", "Light sensitivity", "yn"),
            ("Neck_Stiffness", "Neck stiffness", "yn"),
        ],
    },
    "Abdominal Pain": {
        "flag": "Has_Abdominal_Pain",
        "fields": [
            ("Pain_Location", "Pain location", "painloc"),
            ("Pain_Duration_Level", "Pain duration", "dur"),
            ("Nausea", "Nausea", "yn"),
            ("Diarrhea", "Diarrhea", "yn"),
        ],
    },
    "Fatigue": {
        "flag": "Has_Fatigue",
        "fields": [
            ("Fatigue_Severity", "Fatigue severity", "sev"),
            ("Fatigue_Duration_Level", "Fatigue duration", "dur"),
            ("Weight_Loss", "Weight loss", "yn"),
        ],
    },
    "Vomiting": {
        "flag": "Has_Vomiting",
        "fields": [
            ("Vomiting_Severity", "Vomiting severity", "sev"),
            ("Vomiting_Duration_Level", "Vomiting duration", "dur"),
            ("Blood_Vomit", "Blood in vomit", "yn"),
            ("Unable_To_Keep_Fluids", "Unable to keep fluids down", "yn"),
        ],
    },
}
ALL_FLAGS = [v["flag"] for v in SYMPTOMS.values()]

# ---------------- UI ----------------
st.title("Medical Triage System")
st.markdown("""
**About this system:**
This Medical Triage System is designed to assist healthcare providers in prioritizing patient care based on reported symptoms. 
The system now supports 6 common symptoms: Fever, Cough, Headache, Abdominal Pain, Fatigue, and Vomiting.
Utilizing machine learning, it classifies patients into different urgency levels to help streamline the triage process and ensure timely intervention.

**How to use:**
Select one or more symptoms, then additional questions will appear about the symptoms you selected.
""")

st.caption("If you have symptoms or questions not covered here, please consult a healthcare provider.")

selected = st.multiselect("Symptoms you are experiencing", list(SYMPTOMS.keys()), placeholder="Select symptoms")

# Build payload; default all Has_* to 'maya'
payload = {}
for flag in ALL_FLAGS:
    payload.setdefault(flag, "maya")

# Render follow-ups only for chosen symptoms; set their Has_* to 'haa'
for group in selected:
    cfg = SYMPTOMS[group]
    payload[cfg["flag"]] = "haa"  # user selected this symptom
    with st.expander(group, expanded=True):
        # Create columns for better layout
        if len(cfg["fields"]) <= 2:
            cols = st.columns(len(cfg["fields"]))
        else:
            cols = st.columns(2)  # Max 2 columns for better readability
        
        for i, (col, label, wtype) in enumerate(cfg["fields"]):
            with cols[i % len(cols)]:
                val = render_select(label, wtype, key=f"{group}:{col}")
                if val is not None:
                    payload[col] = val

# Derived feature (fever + fatigue)
if (payload.get("Has_Fever") == "haa") and (payload.get("Has_Fatigue") == "haa"):
    payload["Fever_With_Fatigue"] = "haa"

# Red flags if model expects it
if "Red_Flag_Count" in NUM_COLS:
    def compute_red_flag_count(pl: dict) -> int:
        score = 0
        for k in ["Breath_Difficulty","Blood_Cough","Neck_Stiffness","Blood_Vomit","Unable_To_Keep_Fluids"]:
            if pl.get(k) == "haa":
                score += 1
        for sevk in ["Fever_Severity","Headache_Severity","Fatigue_Severity","Vomiting_Severity"]:
            v = pl.get(sevk) or pl.get(sevk.replace("_Severity","_Level"))
            if v == "aad u daran":
                score += 1
        return score
    payload["Red_Flag_Count"] = compute_red_flag_count(payload)

# ---------------- Predict ----------------
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
if st.button("Assess"):
    if len(selected) == 0:
        st.warning("Please select at least one symptom.")
    else:
        x = make_input_df(payload)
        y_pred = pipe.predict(x)[0]
        label_so = decode_label(y_pred)
        # Convert Somali model output to English for UI display
        label_en = SOMALI_TO_ENGLISH_LABEL.get(label_so, str(label_so))

        # Light, modern result card with dynamic colors
        def triage_style(label_so: str):
            t = (label_so or "").lower()
            if "emergency" in t or "urgent" in t:
                return ("#FFEBEE", "#B71C1C", "#EF9A9A")
            if "moderate" in t or "outpatient" in t:
                return ("#FFF8E1", "#8D6E00", "#FFD54F")
            return ("#E8F5E9", "#1B5E20", "#A5D6A7")
        bg, fg, br = triage_style(label_en)

        st.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:14px;
                background:{bg};
                color:{fg};
                border:1px solid {br};
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
                font-size:1.15rem;
                font-weight:700;
                margin-top:6px;
                margin-bottom:14px;">
                Result: {label_en}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tips card (light blue)
        TRIAGE_TIPS = {
            "Mild condition (Home care)":
                "Rest at home, drink plenty of fluids, eat light meals, take pain relievers or fever reducers if needed, monitor your symptoms for 24 hours, if they worsen contact a healthcare facility.",
            "Moderate condition (Outpatient care)":
                "Visit a healthcare facility within 24 hours for evaluation, bring any previous medication records if available, drink plenty of fluids.",
            "Moderate condition (Outpatient care)":
                "Visit a healthcare facility within 24 hours for evaluation, bring any previous medication records if available, drink plenty of fluids.",
            "Emergency condition":
                "Go to the hospital immediately, do not attempt home treatment, if possible have someone accompany you, bring any previous medication records if available."
        }
        st.markdown(
            """
            <div style="
                padding:16px;
                border-radius:12px;
                background:#E3F2FD;
                color:#0D47A1;
                border:1px solid #90CAF9;
                box-shadow:0 2px 8px rgba(0,0,0,0.03);
                font-size:1.02rem;">
                <strong>Advice:</strong> """ + (TRIAGE_TIPS.get(label_en) or "General advice: if you are concerned about your condition, contact a healthcare facility.") + """
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:12px; color:#374151;'>" + (
            "Important notice: This is a general assessment to help you understand your condition and next steps. "
            "If you are concerned about your condition, contact a healthcare provider."
        ) + "</div>", unsafe_allow_html=True)

# Footer with developer name
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 20px;'>"
    "Developed by <strong>Mohamed Mustaf Ahmed</strong>"
    "</div>", 
    unsafe_allow_html=True
)
