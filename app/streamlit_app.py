
import streamlit as st
import pandas as pd
import joblib
import os
import re

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Smart Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# LOAD MODEL AND VECTORIZER
# -------------------------------

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/model.joblib")

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.joblib"
    )

    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
    artifacts_loaded = True

except Exception as e:
    artifacts_loaded = False
    st.error(f"Could not load model files: {e}")


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'<.*?>', ' ', text)

    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------
# LOAD EVALUATION RESULTS
# -------------------------------

@st.cache_data
def load_results():

    return pd.read_csv(
        "outputs/evaluation_results/model_comparison.csv"
    )


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("📰 Fake News Detector")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "News Detection",
        "Model Performance",
        "Visualizations"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application uses NLP, TF-IDF "
    "and Machine Learning to classify news."
)


# -------------------------------
# HOME PAGE
# -------------------------------

if page == "Home":

    st.title("📰 Smart Fake News Detection System")

    st.subheader(
        "Machine Learning Based News Classification"
    )

    st.write(
        """
        Welcome to the Smart Fake News Detection System.

        This application analyzes news headlines and
        articles using Natural Language Processing
        and Machine Learning.
        """
    )

    st.markdown("### Project Features")

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithms", "2")

    col2.metric("NLP Technique", "TF-IDF")

    col3.metric("Interface", "Streamlit")

    st.markdown("---")

    st.markdown("### How to use")

    st.write(
        """
        1. Open the News Detection section.
        2. Enter a news headline or article.
        3. Click the Detect News button.
        4. View the model prediction.
        """
    )

    st.warning(
        "This system is a text classifier, not a "
        "professional fact-checking service."
    )


# -------------------------------
# NEWS DETECTION PAGE
# -------------------------------

elif page == "News Detection":

    st.title("🔍 News Detection")

    st.write(
        "Enter a news headline or complete article below."
    )

    news_text = st.text_area(
        "Enter News Content",
        height=220,
        placeholder="Paste your news article here..."
    )

    if st.button("Detect News", type="primary"):

        if not artifacts_loaded:

            st.error("Model files are not available.")

        elif not news_text.strip():

            st.warning("Please enter some news text.")

        else:

            cleaned = clean_text(news_text)

            if not cleaned:

                st.warning(
                    "Please enter meaningful news text."
                )

            else:

                # Transform input using saved TF-IDF
                features = vectorizer.transform([cleaned])

                # Generate prediction
                prediction = model.predict(features)[0]

                st.markdown("---")

                st.subheader("Prediction Result")

                if prediction == 0:

                    st.error(
                        "🚨 FAKE NEWS (According to the model)"
                    )

                elif prediction == 1:

                    st.success(
                        "✅ REAL NEWS (According to the model)"
                    )

                # Show model score if available
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        features
                    )[0]

                    class_labels = list(model.classes_)

                    predicted_index = class_labels.index(
                        prediction
                    )

                    score = probabilities[predicted_index]

                    st.metric(
                        "Model Confidence Score",
                        f"{score * 100:.2f}%"
                    )

                    st.caption(
                        "This score reflects the model's "
                        "estimated class probability, not "
                        "the factual certainty of the news."
                    )

                st.info(
                    "The prediction is based on patterns "
                    "learned from the training dataset. "
                    "Always verify important claims using "
                    "reliable independent sources."
                )


# -------------------------------
# MODEL PERFORMANCE PAGE
# -------------------------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    try:

        results = load_results()

        st.subheader("Algorithm Comparison")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        for _, row in results.iterrows():

            st.subheader(row["Model"])

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{row['Accuracy'] * 100:.2f}%"
            )

            col2.metric(
                "Precision",
                f"{row['Precision'] * 100:.2f}%"
            )

            col3.metric(
                "Recall",
                f"{row['Recall'] * 100:.2f}%"
            )

            col4.metric(
                "F1 Score",
                f"{row['F1 Score'] * 100:.2f}%"
            )

    except Exception as e:

        st.error(
            f"Could not load evaluation results: {e}"
        )


# -------------------------------
# VISUALIZATION PAGE
# -------------------------------

elif page == "Visualizations":

    st.title("📈 Model Visualizations")

    image_files = {
        "Label Distribution":
            "outputs/visualizations/label_distribution.png",

        "Text Length Distribution":
            "outputs/visualizations/text_length_distribution.png",

        "Average Word Count":
            "outputs/visualizations/average_word_count.png",

        "Fake News Frequent Words":
            "outputs/visualizations/fake_news_top_words.png",

        "Real News Frequent Words":
            "outputs/visualizations/real_news_top_words.png",

        "Logistic Regression Confusion Matrix":
            "outputs/visualizations/logistic_regression_confusion_matrix.png",

        "Naive Bayes Confusion Matrix":
            "outputs/visualizations/multinomial_naive_bayes_confusion_matrix.png",

        "Model Comparison":
            "outputs/visualizations/model_comparison.png"
    }

    for title, image_path in image_files.items():

        if os.path.exists(image_path):

            st.subheader(title)

            st.image(
                image_path,
                use_container_width=True
            )

        else:

            st.warning(
                f"Visualization not found: {title}"
            )


# -------------------------------
# FOOTER
# -------------------------------

st.markdown("---")

st.caption(
    "Smart Fake News Detection System | "
    "Python | NLP | TF-IDF | Machine Learning | Streamlit"
)
