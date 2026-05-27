import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Mental Health Monitor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD CSS ----------------

def local_css(cssfile):
    with open(cssfile) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

local_css("style.css")

# ---------------- LOAD FILES ----------------

model = load_model(
    "mental_health_rnn_model.h5"
)

with open(
    "tokenizer.pkl",
    "rb"
) as f:

    tokenizer=pickle.load(f)

with open(
    "label_encoder.pkl",
    "rb"
) as f:

    encoder=pickle.load(f)

# ---------------- NLTK ----------------

nltk.download("stopwords")

stop_words=set(
    stopwords.words("english")
)

max_len=100

# ---------------- PREPROCESS ----------------

def preprocess_text(text):

    text=str(text)

    text=text.lower()

    text=re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    words=text.split()

    words=[
        w for w in words
        if w not in stop_words
    ]

    return " ".join(words)

# ---------------- TIPS ----------------

tips={

"Anxiety":
"💙 Take a deep breath. Try a short walk or speak with someone you trust.",

"Depression":
"🌸 Small steps matter. Rest, hydrate, and reach out to supportive people.",

"Stress":
"☕ Take a short break and focus on one task at a time.",

"Suicidal":
"❤️ Please connect with a trusted person or professional support immediately.",

"Bipolar":
"🌙 Sleep, routine and emotional support are important.",

"Normal":
"✨ Keep maintaining your healthy and positive routines.",

"Personality disorder":
"🫶 Practice self-awareness and seek support when needed."
}

# ---------------- HERO ----------------

st.markdown("""

<div class="hero">

<div class="hero-title">
🧠 AI-Based Mental Health Sentiment Monitoring System
</div>

<div class="hero-sub">
Emotion Detection using Simple Recurrent Neural Networks
</div>

<div class="hero-badges">

<div class="badge">
NLP
</div>

<div class="badge">
Simple RNN
</div>

<div class="badge">
Mental Health AI
</div>

<div class="badge">
Emotion Detection
</div>

</div>

</div>

""",unsafe_allow_html=True)

# ---------------- ABOUT ----------------

st.markdown("""

<div class="panel">

<div class="panel-label">
ABOUT PROJECT
</div>

<h2>Why Emotional AI Matters</h2>

<p>

Emotional AI can understand user feelings from text and help monitor emotional well-being.

This system can support:

</p>

<ul>

<li>Sentiment Analysis</li>

<li>Mental Health Monitoring</li>

<li>Counselor Assistance</li>

<li>Early Intervention Support</li>

<li>Emotion-Aware AI Systems</li>

</ul>

<p>

Simple RNN models learn sequence patterns and understand emotional context using hidden states.

</p>

</div>

""",unsafe_allow_html=True)

# ---------------- INPUT ----------------

st.markdown("""

<div class="panel">

<div class="panel-label">
USER INPUT
</div>

<h2>Enter Thoughts</h2>

</div>

""",unsafe_allow_html=True)

user_text=st.text_area(
"Enter your thoughts",
placeholder="Enter your thoughts or feelings here...",
height=180,
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

analyze=st.button(
"Analyze Emotion"
)

# ---------------- PREDICTION ----------------

if analyze:

    if user_text.strip()=="":

        st.warning(
        "Please enter some text."
        )

    else:

        try:

            with st.spinner(
            "Analyzing..."
            ):

                clean=preprocess_text(
                    user_text
                )

                seq=tokenizer.texts_to_sequences(
                    [clean]
                )

                padded=pad_sequences(
                    seq,
                    maxlen=max_len,
                    padding='post'
                )

                pred=model.predict(
                    padded,
                    verbose=0
                )

                probs=pred[0]

                emotion=encoder.inverse_transform(
                    [np.argmax(probs)]
                )[0]

                confidence=float(
                    np.max(probs)*100
                )

# ---------- RESULT ----------

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
unsafe_allow_html=True)

# ---------- METRICS ----------

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
{confidence:.1f}%
</span>

<span class='mt-lbl'>
Confidence
</span>

</div>

<div class='metric-tile'>

<span class='mt-val'>
{"Healthy" if emotion=="Normal" else "Needs Attention"}
</span>

<span class='mt-lbl'>
Status
</span>

</div>

</div>

""",
unsafe_allow_html=True)

# ---------- BARS ----------

            st.markdown(
            "<h3>Emotion Probability</h3>",
            unsafe_allow_html=True
            )

            for cls,prob in zip(
                encoder.classes_,
                probs
            ):

                pct=round(
                    prob*100,
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

# ---------- GUIDANCE ----------

            st.success(
                tips.get(
                    emotion,
                    "Take care of yourself."
                )
            )

# ---------- FOOTER ----------

            st.markdown("""

<div class='info-grid'>

<div class='info-tile'>
<div class='info-title'>
Emotional Monitoring
</div>
<div class='info-body'>
Detect emotional trends from text.
</div>
</div>

<div class='info-tile'>
<div class='info-title'>
AI Sequence Learning
</div>
<div class='info-body'>
RNN learns context from previous words.
</div>
</div>

<div class='info-tile'>
<div class='info-title'>
Wellness Guidance
</div>
<div class='info-body'>
Provides supportive emotional tips.
</div>
</div>

</div>

""",
unsafe_allow_html=True)

        except Exception as e:

            st.error(
                f"Error: {e}"
            )