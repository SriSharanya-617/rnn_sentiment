import streamlit as st
import numpy as np
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Mental Health Monitoring",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD CSS ----------------

def local_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

local_css("style.css")

# ---------------- NLTK ----------------

try:

    nltk.data.find("corpora/stopwords")

except:

    nltk.download("stopwords")

stop_words = set(
    stopwords.words("english")
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mental_health_rnn_model.h5"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

@st.cache_resource
def load_all():

    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "SimpleRNN": SimpleRNN
        }
    )

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    with open(
        ENCODER_PATH,
        "rb"
    ) as f:

        encoder = pickle.load(f)

    return model, tokenizer, encoder


try:

    model, tokenizer, encoder = load_all()

except Exception as e:

    st.error(
        f"Error loading model: {e}"
    )

    st.stop()


# ---------------- SETTINGS ----------------

max_len = 100


# ---------------- PREPROCESS ----------------

def preprocess_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    words = text.split()

    words = [
        w
        for w in words
        if w not in stop_words
    ]

    return " ".join(words)


# ---------------- TIPS ----------------

tips = {

    "Anxiety":
    "💙 Slow down. Take deep breaths and speak with someone you trust.",

    "Depression":
    "🌸 Small steps matter. Take a short walk and talk with loved ones.",

    "Stress":
    "☕ Take a break and focus on one task at a time.",

    "Suicidal":
    "❤️ Please reach out to trusted people or professional help immediately.",

    "Bipolar":
    "🌙 Maintain sleep routines and stay connected with support systems.",

    "Normal":
    "✨ Great! Keep maintaining healthy habits.",

    "Personality disorder":
    "🫶 Self awareness and support systems can help."
}


# ---------------- HERO ----------------

st.markdown("""

<div class='hero'>

<div class='hero-title'>

🧠 AI-Based Mental Health Sentiment Monitoring System

</div>

<div class='hero-sub'>

Emotion Detection using Simple Recurrent Neural Networks

</div>

<div class='hero-badges'>

<div class='badge'>
NLP
</div>

<div class='badge'>
Simple RNN
</div>

<div class='badge'>
AI
</div>

<div class='badge'>
Mental Health
</div>

</div>

</div>

""",
unsafe_allow_html=True
)


# ---------------- ABOUT ----------------

st.markdown("""

<div class='panel'>

<div class='panel-label'>

ABOUT PROJECT

</div>

<h2>

Why Emotional AI Matters

</h2>

<p>

Emotional AI can understand user feelings from text and help monitor emotional well-being.

</p>

<p>

This system supports:

</p>

<ul>

<li>Sentiment Analysis</li>

<li>Mental Health Monitoring</li>

<li>Counselor Assistance</li>

<li>Early Intervention</li>

<li>Emotion-Aware Systems</li>

</ul>

<p>

Simple RNN models learn sequence patterns and hidden-state context.

</p>

</div>

""",
unsafe_allow_html=True
)


# ---------------- INPUT ----------------

st.markdown("""

<div class='panel'>

<div class='panel-label'>

USER INPUT

</div>

<h2>

Enter Thoughts

</h2>

</div>

""",
unsafe_allow_html=True
)


user_text = st.text_area(
    "",
    placeholder=
    "Enter your thoughts or feelings here...",
    height=200,
    label_visibility="collapsed"
)

st.info("""

Examples:

• I feel anxious and unable to sleep

• I am happy and enjoying my day

• I feel lonely and depressed

• I feel stressed about my future

""")


# ---------------- BUTTON ----------------

analyze = st.button(
    "Analyze Emotion"
)


# ---------------- PREDICTION ----------------

if analyze:

    if user_text.strip() == "":

        st.warning(
            "Please enter text"
        )

    else:

        with st.spinner(
            "Analyzing..."
        ):

            clean = preprocess_text(
                user_text
            )

            seq = tokenizer.texts_to_sequences(
                [clean]
            )

            padded = pad_sequences(
                seq,
                maxlen=max_len,
                padding='post'
            )

            pred = model.predict(
                padded,
                verbose=0
            )

            probs = pred[0]

            emotion = encoder.inverse_transform(
                [np.argmax(probs)]
            )[0]

            confidence = np.max(
                probs
            ) * 100

            status = (
                "Healthy"
                if emotion == "Normal"
                else
                "Needs Attention"
            )

        # RESULT

        st.markdown(

f"""

<div class='verdict'>

<div class='verdict-title'>

Emotion Detected:
{emotion}

</div>

<div class='verdict-detail'>

Confidence:
{confidence:.2f}%

</div>

</div>

""",

unsafe_allow_html=True

)

        # METRICS

        st.markdown(

f"""

<div class='metric-row'>

<div class='metric-tile'>

<span class='mt-val'>
{emotion}
</span>

<span class='mt-lbl'>
Prediction
</span>

</div>


<div class='metric-tile'>

<span class='mt-val'>
{confidence:.2f}%
</span>

<span class='mt-lbl'>
Confidence
</span>

</div>


<div class='metric-tile'>

<span class='mt-val'>
{status}
</span>

<span class='mt-lbl'>
Status
</span>

</div>

</div>

""",

unsafe_allow_html=True

)

        # BARS

        st.markdown(
            "<h3>Emotion Probability</h3>",
            unsafe_allow_html=True
        )

        for cls, prob in zip(
            encoder.classes_,
            probs
        ):

            pct = round(
                prob * 100,
                1
            )

            st.markdown(

f"""

<div class='bar-row'>

<div class='bar-name'>

{cls}

</div>

<div class='bar-track'>

<div
class='bar-fill'
style='width:{pct}%'>

</div>

</div>

<div class='bar-pct'>

{pct}%

</div>

</div>

""",

unsafe_allow_html=True
)

        # TIPS

        st.success(
            tips.get(
                emotion,
                "Take care of yourself"
            )
        )