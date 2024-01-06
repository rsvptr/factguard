# Libraries required by the webapp.
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import model_helperfunctions
import nltk

nltk.download('punkt')

# Sets the page title for the webapp.
st.set_page_config(
        page_title="FactGuard",
)

# Sets the background image for the webapp.
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1637167473291-9f8caf792ded?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2404&q=95");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# Setting value of constants.
MODEL_PATH = './Models/'
MODEL_FILE_NAME = 'rf_tf-idf_plusguardian_model.sav'
RANDOM_STATE = 42
DATA_PATH = './Data/'

# Specifying the local model helper function and the stopwords list.
gist_file = open(DATA_PATH + "gist_stopwords.txt", "r")
try:
    content = gist_file.read()
    expanded_stopwords = content.split(",")
finally:
    gist_file.close()

expanded_stopwords.remove('via')
expanded_stopwords.remove('eu')
expanded_stopwords.remove('uk')

def lowercase_and_only_expanded_stopwords(doc):
    # Remove stopwords and lowercase tokens.
    stop_words = expanded_stopwords
    return [token.lower() for token in doc if token.lower() in stop_words]

# Loads the pipeline.
@st.cache(allow_output_mutation=True)
def load_pipeline(model_path=MODEL_PATH, model_file_name=MODEL_FILE_NAME):
    
    #Loads the text processing and classifier pipeline.
    
    return pickle.load(open(model_path + model_file_name, 'rb'))

pipeline = load_pipeline()


st.title('FactGuard 📰🛡️')

st.write("""On providing the title and body text of a news article, this application can classify it as truthful or fake with a confidence score (in %). Note that the algorithm is not fact-checking the article. It bases the classification entirely on the style of the title and body text of the provided article.""")

news_title = st.text_input('Enter the title of a news article below:')

if news_title:
    news_story = st.text_area('Enter the body text from a news article below:', height=400)

    if news_story and news_title:
        tokens = model_helperfunctions.tokenize_and_normalize_title_and_text(news_title, news_story)
        stop_words_only = lowercase_and_only_expanded_stopwords(tokens)
        if len(stop_words_only) == 0:
            st.write('There were no stopwords detected in your article title and/or body.')
        else:
            class_ = pipeline.predict([tokens])
            if class_ == 0:
                class_text = 'fake'
            else:
                class_text = 'truthful'

            probability = round(pipeline.predict_proba([tokens])[0][class_][0] * 100, 2)
            st.subheader('Classification Results')
            st.write('The given news article is classified as ', class_text, 'with a confidence score of',
                     probability, '%.')
            st.write()
            st.subheader('The article represented only as stopwords:')
            st.write(' '.join(stop_words_only))
