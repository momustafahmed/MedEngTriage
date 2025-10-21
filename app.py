import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import json

# ---------------- Basic setup ----------------
st.set_page_config(page_title="Talo bixiye Caafimaad", layout="centered")

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

# --------------- Choices (Somali) ---------------
YN = ["haa", "maya"]
SEV = ["fudud", "dhexdhexaad", "aad u daran"]
COUGH_TYPE = ["qalalan", "qoyan"]
PAIN_LOC = ["caloosha sare", "caloosha hoose", "caloosha oo dhan"]
AGE_GROUP = ["caruur", "qof weyn", "waayeel"]

# Duration mapping: show phrases, map to model tokens
DUR_TOKEN_TO_DISPLAY = {
    "fudud": "hal maalin iyo ka yar",
    "dhexdhexaad": "labo illaa sadax maalin",
    "dhexdhexaad ah": "labo illaa sadax maalin",
    "aad u daran": "sadax maalin iyo ka badan",
}
# When user picks a phrase, convert back to token for model input
DUR_DISPLAY_TO_TOKEN = {
    v: ("dhexdhexaad" if k.startswith("dhexdhexaad") else k)
    for k, v in DUR_TOKEN_TO_DISPLAY.items()
}
DUR_DISPLAY = list(dict.fromkeys(DUR_TOKEN_TO_DISPLAY.values()))

# --------------- Default one-sentence tips ---------------
TRIAGE_TIPS = {
    "Xaalad fudud (Daryeel guri)":
        "Ku naso guriga, cab biyo badan, cun cunto fudud, qaado xanuun baabi'iye ama qandho dajiye haddii aad u baahantahay, la soco calaamadahaaga 24 saac, haddii ay kasii daraan la xiriir xarun caafimaad.",
    "Xaalad dhax dhaxaad eh (Bukaan socod)":
        "Booqo xarun caafimaad 24 saacadood gudahood si lagu qiimeeyo, qaado warqadaha daawooyinkii hore haddii ay jiraan, cab biyo badan.",
    "Xaalad dhax dhaxaad ah (Bukaan socod)":
        "Booqo xarun caafimaad 24 saacadood gudahood si lagu qiimeeyo, qaado warqadaha daawooyinkii hore haddii ay jiraan, cab biyo badan.",
    "Xaalad deg deg ah":
        "Si deg deg ah u gaar isbitaalka, ha isku dayin daaweynta guriga, haddii ay suurtagal tahay raac qof kugu weheliya, qaado warqadaha daawooyinkii hore haddii ay jiraan."
}
EXTRA_NOTICE = (
    "Farriin gaar ah: Tan waa qiimeyn guud oo kaa caawinaysa inaad fahanto xaaladdaada iyo waxa xiga. "
    "Haddii aad ka welwelsan tahay xaaladdaada, la xiriir dhakhtar."
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
    if "deg deg" in t:
        return ("#FFEBEE", "#B71C1C", "#EF9A9A")
    if "dhax dhaxaad" in t:
        return ("#FFF8E1", "#8D6E00", "#FFD54F")
    return ("#E8F5E9", "#1B5E20", "#A5D6A7")

def render_select(label, wtype, key):
    placeholder = "Dooro"
    if wtype == "yn":
        return st.selectbox(label, YN, index=None, placeholder=placeholder, key=key)
    if wtype == "sev":
        return st.selectbox(label, SEV, index=None, placeholder=placeholder, key=key)
    if wtype == "cough":
        return st.selectbox(label, COUGH_TYPE, index=None, placeholder=placeholder, key=key)
    if wtype == "painloc":
        return st.selectbox(label, PAIN_LOC, index=None, placeholder=placeholder, key=key)
    if wtype == "dur":
        disp = st.selectbox(label, DUR_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        return DUR_DISPLAY_TO_TOKEN.get(disp, disp)
    return None

# --------------- Symptom groups (Somali-only, NO Has_* question in UI) ---------------
SYMPTOMS = {
    "Qandho": {
        "flag": "Has_Fever",
        "fields": [
            ("Fever_Level", "Heerka qandhada", "sev"),
            ("Fever_Duration_Level", "Mudada qandhada", "dur"),
            ("Chills", "Qarqaryo", "yn"),
        ],
    },
    "Qufac": {
        "flag": "Has_Cough",
        "fields": [
            ("Cough_Type", "Nuuca qufaca", "cough"),
            ("Cough_Duration_Level", "Mudada qufaca", "dur"),
            ("Blood_Cough", "Qufac dhiig", "yn"),
            ("Breath_Difficulty", "Neef qabasho", "yn"),
        ],
    },
    "Madax-xanuun": {
        "flag": "Has_Headache",
        "fields": [
            ("Headache_Severity", "Heerka madax-xanuunka", "sev"),
            ("Headache_Duration_Level", "Mudada madax-xanuunka", "dur"),
            ("Photophobia", "Iftiinka ku dhibaya", "yn"),
            ("Neck_Stiffness", "Qoor adkaaday", "yn"),
        ],
    },
    "Calool-xanuun": {
        "flag": "Has_Abdominal_Pain",
        "fields": [
            ("Pain_Location", "Goobta xanuunka caloosha", "painloc"),
            ("Pain_Duration_Level", "Mudada xanuunka caloosha", "dur"),
            ("Nausea", "Lallabbo", "yn"),
            ("Diarrhea", "Shuban", "yn"),
        ],
    },
    "Daal": {
        "flag": "Has_Fatigue",
        "fields": [
            ("Fatigue_Severity", "Heerka daalka", "sev"),
            ("Fatigue_Duration_Level", "Mudada daalka", "dur"),
            ("Weight_Loss", "Miisaan dhimista", "yn"),
        ],
    },
    "Matag": {
        "flag": "Has_Vomiting",
        "fields": [
            ("Vomiting_Severity", "Heerka matagga", "sev"),
            ("Vomiting_Duration_Level", "Mudada matagga", "dur"),
            ("Blood_Vomit", "Matag dhiig", "yn"),
            ("Unable_To_Keep_Fluids", "Aan ceshan karin dareeraha", "yn"),
        ],
    },
}
ALL_FLAGS = [v["flag"] for v in SYMPTOMS.values()]

# ---------------- UI ----------------
st.title("Talo bixiye Caafimaad")
st.markdown("Dooro hal calaamad ama wax ka badan, ka dibna waxaa kuusoo muuqan doono su'aalo dheeraad ah oo ku saabsan calaamadaha aad dooratay.")

st.caption("Haddii ay jiraan calaamado ama su'aalo aan ku khusayn, ka gudub.")

selected = st.multiselect("Calaamadaha aad qabto", list(SYMPTOMS.keys()), placeholder="Dooro calaamad")

# Build payload; default all Has_* to 'maya'
payload = {}
for flag in ALL_FLAGS:
    payload.setdefault(flag, "maya")

# Render follow-ups only for chosen symptoms; set their Has_* to 'haa'
for group in selected:
    cfg = SYMPTOMS[group]
    payload[cfg["flag"]] = "haa"  # user selected this symptom
    with st.expander(group, expanded=True):
        for (col, label, wtype) in cfg["fields"]:
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
if st.button("Qiimee"):
    if len(selected) == 0:
        st.warning("Fadlan dooro ugu yaraan hal calaamad.")
    else:
        x = make_input_df(payload)
        y_pred = pipe.predict(x)[0]
        label_so = decode_label(y_pred)

        # Light, modern result card with dynamic colors
        def triage_style(label_so: str):
            t = (label_so or "").lower()
            if "deg deg" in t:
                return ("#FFEBEE", "#B71C1C", "#EF9A9A")
            if "dhax dhaxaad" in t:
                return ("#FFF8E1", "#8D6E00", "#FFD54F")
            return ("#E8F5E9", "#1B5E20", "#A5D6A7")
        bg, fg, br = triage_style(label_so)

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
                Natiijada: {label_so}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tips card (light blue)
        TRIAGE_TIPS = {
            "Xaalad fudud (Daryeel guri)":
                "Ku naso guriga, cab biyo badan, cun cunto fudud, qaado xanuun baabi'iye ama qandho dajiye haddii aad u baahantahay, la soco calaamadahaaga 24 saac, haddii ay kasii daraan la xiriir xarun caafimaad.",
            "Xaalad dhax dhaxaad eh (Bukaan socod)":
                "Booqo xarun caafimaad 24 saacadood gudahood si lagu qiimeeyo, qaado warqadaha daawooyinkii hore haddii ay jiraan, cab biyo badan.",
            "Xaalad dhax dhaxaad ah (Bukaan socod)":
                "Booqo xarun caafimaad 24 saacadood gudahood si lagu qiimeeyo, qaado warqadaha daawooyinkii hore haddii ay jiraan, cab biyo badan.",
            "Xaalad deg deg ah":
                "Si deg deg ah u gaar isbitaalka, ha isku dayin daaweynta guriga, haddii ay suurtagal tahay raac qof kugu weheliya, qaado warqadaha daawooyinkii hore haddii ay jiraan."
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
                <strong>Talo:</strong> """ + (TRIAGE_TIPS.get(label_so) or "La-talin guud: haddii aad ka welwelsan tahay xaaladdaada, la xiriir xarun caafimaad.") + """
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:12px; color:#374151;'>" + (
            "Farriin gaar ah: Tan waa qiimeyn guud oo kaa caawinaysa inaad fahanto xaaladdaada iyo waxa xiga. "
            "Haddii aad ka welwelsan tahay xaaladdaada, la xiriir dhakhtar."
        ) + "</div>", unsafe_allow_html=True)
