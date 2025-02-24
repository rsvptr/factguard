import streamlit as st
import pickle
import pandas as pd
import numpy as np
import model_helperfunctions
import nltk

# Download required NLTK resources quietly.
nltk.download('punkt', quiet=True)

# Configure the Streamlit page.
st.set_page_config(
    page_title="FactGuard",
    layout="centered"
)

# Add background image.
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1637167473291-9f8caf792ded?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2404&q=95");
            background-attachment: fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_from_url()

# Constants.
MODEL_PATH = './Models/'
MODEL_FILE_NAME = 'rf_tf-idf_plusguardian_model.sav'
DATA_PATH = './Data/'

# Load expanded stopwords from file using a context manager.
with open(DATA_PATH + "gist_stopwords.txt", "r") as gist_file:
    content = gist_file.read()
expanded_stopwords = content.split(",")
# Remove undesired stopwords.
for word in ['via', 'eu', 'uk']:
    if word in expanded_stopwords:
        expanded_stopwords.remove(word)

def lowercase_and_only_expanded_stopwords(doc):
    """
    Return a list of tokens that are lowercased and exist in the expanded stopwords.
    """
    return [token.lower() for token in doc if token.lower() in expanded_stopwords]

# Load the model pipeline using Streamlit's new caching method.
@st.cache_resource
def load_pipeline(model_path=MODEL_PATH, model_file_name=MODEL_FILE_NAME):
    """Load the pickled text processing and classification pipeline."""
    with open(model_path + model_file_name, 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

pipeline = load_pipeline()

# App title and description.
st.title('FactGuard 📰🛡️')
st.write("""
Provide the title and body text of a news article to classify it as truthful or fake, along with a confidence score.
*Note:* The classification is based solely on stylistic features and does not perform a fact check.
""")

# Input for news article title and body.
news_title = st.text_input('Enter the title of a news article:')
news_story = st.text_area('Enter the body text of the news article:', height=400)

if news_title and news_story:
    # Tokenize and normalize the input.
    tokens = model_helperfunctions.tokenize_and_normalize_title_and_text(news_title, news_story)
    stop_words_only = lowercase_and_only_expanded_stopwords(tokens)

    if not stop_words_only:
        st.error('No stopwords detected in the provided article title or body.')
    else:
        # Predict the class (0: fake, 1: truthful).
        predicted_class = pipeline.predict([tokens])[0]
        class_text = 'truthful' if predicted_class else 'fake'

        # Retrieve the probability for the predicted class.
        probability = round(pipeline.predict_proba([tokens])[0][predicted_class] * 100, 2)

        st.subheader('Classification Results')
        st.write(f'The news article is classified as **{class_text}** with a confidence score of **{probability}%**.')
        st.subheader('Extracted Stopwords from the Article')
        st.write(' '.join(stop_words_only))
