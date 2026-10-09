
import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cyberbullying Classification",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# TEXT PREPROCESSING
# Harus sama dengan preprocessing saat training
# ============================================================

def clean_text(text):

    # Pastikan input berupa string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Menghapus URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Menghapus mention/username
    text = re.sub(r"@\w+", " ", text)

    # Menghapus marker RT
    text = re.sub(r"\brt\b", " ", text)

    # Menghilangkan simbol hashtag,
    # tetapi mempertahankan kata setelah #
    text = text.replace("#", "")

    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r"[^a-z\s]", " ", text)

    # Menghapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD MODEL
# ============================================================

from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():

    model = joblib.load(
        BASE_DIR / "cyberbullying_model.joblib"
    )

    tfidf_word = joblib.load(
        BASE_DIR / "tfidf_word_vectorizer.joblib"
    )

    tfidf_char = joblib.load(
        BASE_DIR / "tfidf_char_vectorizer.joblib"
    )

    return model, tfidf_word, tfidf_char


model, tfidf_word, tfidf_char = load_model()

# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Cyberbullying Classification")

st.write(
    """
    Aplikasi ini menggunakan Machine Learning untuk
    mengklasifikasikan teks ke dalam enam kategori
    cyberbullying.
    """
)


# ============================================================
# USER INPUT
# ============================================================

user_text = st.text_area(
    "Masukkan teks yang ingin dianalisis:",
    height=150,
    placeholder="Contoh: You are such an ugly stupid girl"
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Analyze Text"):

    if not user_text.strip():

        st.warning("Silakan masukkan teks terlebih dahulu.")

    else:

        # -------------------------
        # Preprocessing
        # -------------------------

        cleaned_text = clean_text(user_text)

        # -------------------------
        # TF-IDF Transformation
        # -------------------------

        word_features = tfidf_word.transform(
            [cleaned_text]
        )

        char_features = tfidf_char.transform(
            [cleaned_text]
        )

        combined_features = hstack([
            word_features,
            char_features
        ])

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(
            combined_features
        )[0]

        probabilities = model.predict_proba(
            combined_features
        )[0]

        # -------------------------
        # RESULT
        # -------------------------

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: **{prediction}**"
        )

        # Confidence
        max_probability = probabilities.max()

        st.metric(
            "Confidence",
            f"{max_probability * 100:.2f}%"
        )

        # -------------------------
        # CLEANED TEXT
        # -------------------------

        with st.expander(
            "View Preprocessed Text"
        ):

            st.write(cleaned_text)

        # -------------------------
        # CLASS PROBABILITIES
        # -------------------------

        st.subheader(
            "Probability by Class"
        )

        probability_data = {
            label: float(prob)
            for label, prob
            in zip(
                model.classes_,
                probabilities
            )
        }

        probability_data = dict(
            sorted(
                probability_data.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        for label, prob in probability_data.items():

            st.write(
                f"**{label}** — {prob*100:.2f}%"
            )

            st.progress(
                float(prob)
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About the Model"):

    st.write(
        """
        **Algorithm:** Logistic Regression

        **Text Representation:** Word TF-IDF + Character TF-IDF

        **Word Features:** 10,000

        **Character Features:** 10,000

        **Total Features:** 20,000

        **Regularization Parameter (C):** 0.25

        **Test Accuracy:** 83.53%

        **Macro F1-score:** 83.38%
        """
    )

    st.caption(
        "Prediksi merupakan hasil dari model machine learning "
        "dan tidak selalu menunjukkan klasifikasi yang benar."
    )
