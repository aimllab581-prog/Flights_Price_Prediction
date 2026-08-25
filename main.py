import re

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from textwrap import dedent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkyFare | Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# ARTIFACT PATHS  -- must match what your training script saved
# ============================================================

MODEL_PATH = "Flights_price_predict.pkl"
ENCODER_PATH = "Flights_encoding.pkl"

# Order these were fit in (this is exactly what
# df.select_dtypes(include=['object', 'category']).columns
# produced from your training dataframe -- categorical columns
# keep their original left-to-right order, they aren't sorted
# alphabetically). Getting this order wrong will silently feed
# the wrong category into the wrong column.
CATEGORICAL_COLS = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]

# Full feature order the model was trained on (X = df.iloc[:, :-1]).
FEATURE_ORDER = CATEGORICAL_COLS + ["duration", "days_left"]


# ============================================================
# HELPER FOR HTML
# ============================================================

def render_html(content):
    """
    Dedents and collapses blank lines before handing the string to
    st.markdown. Blank lines inside a raw HTML block make Streamlit's
    markdown parser drop out of "HTML mode" partway through, which
    causes tags to render as literal text instead of real HTML.
    """
    content = dedent(content)
    content = re.sub(r"\n\s*\n+", "\n", content).strip("\n")
    st.markdown(content, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS  -- sky / aviation theme
# ============================================================

render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
        --sky: #0ea5e9;
        --sky-deep: #0369a1;
        --navy: #0f2540;
        --amber: #f59e0b;
        --paper: #ffffff;
        --bg: #f3f9fd;
        --border: #dbe9f3;
        --text: #10233a;
        --text-soft: #4c6178;
        --text-faint: #8ba0b3;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 1rem !important;
        max-width: 760px !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }
    div[data-testid="stToolbar"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .stApp {
        background:
            radial-gradient(circle at 8% 6%, rgba(14, 165, 233, 0.14), transparent 30%),
            radial-gradient(circle at 92% 10%, rgba(245, 158, 11, 0.10), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(14, 165, 233, 0.08), transparent 34%),
            var(--bg);
    }

    /* ---------------- HERO ---------------- */
    .hero-wrapper {
        text-align: center !important;
        padding: 32px 20px 20px;
    }
    .logo-lockup {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .logo-emoji {
        font-size: 38px;
        filter: drop-shadow(0 6px 12px rgba(14, 165, 233, 0.30));
    }
    .hero-title {
        margin: 0;
        font-family: 'Playfair Display', serif;
        font-size: clamp(38px, 6vw, 56px);
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, var(--sky-deep), var(--sky), var(--amber));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle,
    div[data-testid="stMarkdownContainer"] .hero-subtitle {
        text-align: center !important;
        max-width: 460px;
        margin: 12px auto 0;
        color: var(--text-soft);
        font-size: 14.5px;
        line-height: 1.6;
    }

    /* ---------------- SECTION LABELS ---------------- */
    .section-label {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 26px 0 10px;
        font-size: 12.5px;
        font-weight: 800;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: var(--sky-deep);
    }
    .section-label:first-of-type { margin-top: 8px; }

    /* ---------------- FORM WIDGETS ---------------- */
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1.5px solid var(--border) !important;
        background: var(--paper) !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: var(--sky) !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSlider"] {
        border-radius: 12px;
    }
    div[data-testid="stNumberInput"] input {
        border: 1.5px solid var(--border) !important;
        border-radius: 12px !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: var(--sky) !important;
    }

    div.stButton > button, div.stFormSubmitButton > button {
        width: 100%;
        border-radius: 14px !important;
        border: none !important;
        background: linear-gradient(90deg, var(--sky-deep), var(--sky)) !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 15px !important;
        padding: 12px 0 !important;
        box-shadow: 0 10px 24px rgba(14, 165, 233, 0.28) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(14, 165, 233, 0.35) !important;
    }

    /* ---------------- RESULT CARD ---------------- */
    .result-card {
        margin-top: 26px;
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(135deg, var(--navy), var(--sky-deep));
        text-align: center;
        box-shadow: 0 18px 40px rgba(15, 37, 64, 0.25);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: "✈️";
        position: absolute;
        font-size: 90px;
        opacity: 0.08;
        top: -14px;
        right: 6px;
        transform: rotate(18deg);
    }
    .result-label {
        color: var(--amber);
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .result-price {
        font-family: 'Playfair Display', serif;
        color: white;
        font-size: 44px;
        font-weight: 800;
    }
    .result-route {
        margin-top: 10px;
        color: rgba(255,255,255,0.75);
        font-size: 13.5px;
        font-weight: 600;
    }

    /* ---------------- SUMMARY CHIPS ---------------- */
    .chip-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 16px;
    }
    .chip {
        padding: 6px 12px;
        border-radius: 50px;
        background: rgba(255,255,255,0.10);
        color: white;
        font-size: 11.5px;
        font-weight: 700;
    }

    /* ---------------- WARNING ---------------- */
    .warn-box {
        margin-top: 16px;
        padding: 14px 16px;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fde3c7;
        color: #9a5b0a;
        font-size: 13.5px;
        font-weight: 600;
        text-align: center;
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid var(--border);
        color: var(--text-faint);
        font-size: 12px;
    }

    @media (max-width: 768px) {
        .result-price { font-size: 34px; }
    }
</style>
""")


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero-wrapper">
    <div class="logo-lockup">
        <span class="logo-emoji">🛫</span>
        <h1 class="hero-title">SkyFare</h1>
    </div>
    <p class="hero-subtitle">
        Pick your route and travel details, and the model
        will estimate the ticket price before you book.
    </p>
</div>
""")


# ============================================================
# LOAD MODEL + ENCODER
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


try:
    model, encoder = load_artifacts()
except FileNotFoundError as e:
    render_html(f"""
    <div class="warn-box">
        Couldn't find <b>{MODEL_PATH}</b> or <b>{ENCODER_PATH}</b> in this
        folder. Make sure both files sit next to this script.<br>({e})
    </div>
    """)
    st.stop()

# Pull the exact category lists the encoder was fit on, so the
# dropdowns can never offer a value the model has never seen.
CATEGORY_OPTIONS = {
    col: list(cats) for col, cats in zip(CATEGORICAL_COLS, encoder.categories_)
}


def pretty(label: str) -> str:
    return label.replace("_", " ").strip().title()


# ============================================================
# FORM
# ============================================================

with st.form("flight_form"):

    render_html('<div class="section-label">🛫 Route</div>')
    col1, col2 = st.columns(2)
    with col1:
        source_city = st.selectbox(
            "From",
            options=CATEGORY_OPTIONS["source_city"],
            format_func=pretty
        )
    with col2:
        destination_city = st.selectbox(
            "To",
            options=CATEGORY_OPTIONS["destination_city"],
            format_func=pretty
        )

    render_html('<div class="section-label">🛩️ Airline &amp; Class</div>')
    col3, col4 = st.columns(2)
    with col3:
        airline = st.selectbox(
            "Airline",
            options=CATEGORY_OPTIONS["airline"],
            format_func=pretty
        )
    with col4:
        travel_class = st.selectbox(
            "Class",
            options=CATEGORY_OPTIONS["class"],
            format_func=pretty
        )

    render_html('<div class="section-label">🕐 Timing</div>')
    col5, col6, col7 = st.columns(3)
    with col5:
        departure_time = st.selectbox(
            "Departure",
            options=CATEGORY_OPTIONS["departure_time"],
            format_func=pretty
        )
    with col6:
        arrival_time = st.selectbox(
            "Arrival",
            options=CATEGORY_OPTIONS["arrival_time"],
            format_func=pretty
        )
    with col7:
        stops = st.selectbox(
            "Stops",
            options=CATEGORY_OPTIONS["stops"],
            format_func=pretty
        )

    render_html('<div class="section-label">⏱️ Duration &amp; Booking Window</div>')
    col8, col9 = st.columns(2)
    with col8:
        duration = st.number_input(
            "Flight duration (hours)",
            min_value=0.5,
            max_value=50.0,
            value=2.5,
            step=0.25,
            help="Total travel time, including layovers, in hours."
        )
    with col9:
        days_left = st.slider(
            "Days left until departure",
            min_value=1,
            max_value=50,
            value=15
        )

    submitted = st.form_submit_button("✈️  Predict Price")


# ============================================================
# PREDICT
# ============================================================

if submitted:
    if source_city == destination_city:
        render_html("""
        <div class="warn-box">
            ⚠️ Source and destination city are the same -- pick two
            different cities to get a real prediction.
        </div>
        """)
    else:
        raw_row = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": travel_class,
        }])[CATEGORICAL_COLS]  # enforce fit-time column order

        encoded_cats = encoder.transform(raw_row)[0]

        feature_row = list(encoded_cats) + [duration, days_left]
        X_input = pd.DataFrame([feature_row], columns=FEATURE_ORDER)

        predicted_price = model.predict(X_input)[0]

        stop_label = pretty(stops)

        render_html(f"""
        <div class="result-card">
            <div class="result-label">Estimated Fare</div>
            <div class="result-price">₹{predicted_price:,.0f}</div>
            <div class="result-route">
                {pretty(source_city)} → {pretty(destination_city)}
                &nbsp;&middot;&nbsp; {pretty(airline)}
            </div>
            <div class="chip-strip">
                <div class="chip">🛫 {pretty(departure_time)}</div>
                <div class="chip">🛬 {pretty(arrival_time)}</div>
                <div class="chip">🔁 {stop_label}</div>
                <div class="chip">💺 {pretty(travel_class)}</div>
                <div class="chip">⏱️ {duration:g}h</div>
                <div class="chip">🗓️ {days_left}d left</div>
            </div>
        </div>
        """)


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">
    🛫 SkyFare &nbsp;&middot;&nbsp; Decision Tree Flight Price Predictor
</div>
""")