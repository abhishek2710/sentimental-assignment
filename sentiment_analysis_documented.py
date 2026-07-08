import streamlit as st
import joblib
import re
import string
import nltk
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Sentiment Analysis using Multiple Models")

st.markdown(
"""
Choose a Machine Learning model from the sidebar and
predict the sentiment of a product review.
"""
)

# -------------------------------------------------------
# DOWNLOAD NLTK DATA ONLY IF REQUIRED
# -------------------------------------------------------

@st.cache_resource
def setup_nltk():

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\\S+|www\\S+", "", text)

    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------

@st.cache_resource
def load_models():

    model_dir = Path("models")

    models = {

        "Logistic Regression":
            joblib.load(model_dir / "logistic_model.pkl"),

        "Naive Bayes":
            joblib.load(model_dir / "nb_model.pkl"),

        "Random Forest":
            joblib.load(model_dir / "rf_model.pkl"),

        "SVM":
            joblib.load(model_dir / "svm_model.pkl"),

        "XGBoost":
            joblib.load(model_dir / "xgb_model.pkl"),

        "ANN":
            joblib.load(model_dir / "ann_model.pkl")

    }

    tfidf = joblib.load(model_dir / "tfidf.pkl")

    encoder = joblib.load(model_dir / "label_encoder.pkl")

    vader = SentimentIntensityAnalyzer()

    return models, tfidf, encoder, vader


models, tfidf, encoder, vader = load_models()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("Model Selection")

selected_model = st.sidebar.selectbox(

    "Choose Model",

    [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "SVM",
        "VADER",
        "XGBoost",
        "ANN"
    ]

)

# -------------------------------------------------------
# INPUT
# -------------------------------------------------------

review = st.text_area(

    "Enter Review",

    placeholder="Example: This mobile phone is really good..."

)

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        cleaned = clean_text(review)

        vector = tfidf.transform([cleaned])

        if selected_model == "VADER":

            score = vader.polarity_scores(review)

            if score["compound"] >= 0.05:
                prediction = "positive"

            elif score["compound"] <= -0.05:
                prediction = "negative"

            else:
                prediction = "neutral"

            confidence = abs(score["compound"])

        else:

            model = models[selected_model]

            if selected_model == "XGBoost":

                pred = model.predict(vector)

                prediction = encoder.inverse_transform(pred)[0]

            else:

                prediction = model.predict(vector)[0]

            confidence = None

            if hasattr(model, "predict_proba"):

                confidence = model.predict_proba(vector).max()
                st.markdown("### Prediction Result")
                st.info(f"**Model Used:** {selected_model}")
                st.success(f"**Prediction:** {prediction.upper()}")
            else:
                st.warning("Please enter a review.")

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

# -------------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.write("### Available Models")

st.sidebar.write("""
- Logistic Regression
- Naive Bayes
- Random Forest
- SVM
- VADER
- XGBoost
- ANN
""")

st.sidebar.markdown("---")

st.sidebar.info(
"""
This application loads pre-trained models.
No model training occurs during startup,
making the application much faster.
"""
)
