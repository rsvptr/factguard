# This file contains various helper functions that are used in the modelling phase. Instead of defining them directly in the notebook, it can be imported and used, which reduces clutter.

from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize


def passthrough(doc):
  
    # Passthrough function for use in the pipeline in case the text is already tokenized.
    return doc

def confustion_matrix_and_classification_report(estimator, X, y, labels, set_name):
    
    # Displays a classfication report and confusion matrix for a given input data.
    predictions = estimator.predict(X)

    print(f'Classification report for {set_name} set')
    print(classification_report(y, predictions, target_names=labels))

    matrix = plot_confusion_matrix(estimator,
                                   X,
                                   y,
                                   display_labels = labels,
                                   cmap = plt.cm.Blues,
                                   xticks_rotation = 70,
                                   values_format = 'd')
    matrix.ax_.set_title(f'{set_name} set confustion matrix, without normalization')

    plt.show()

    matrix = plot_confusion_matrix(estimator,
                                   X,
                                   y,
                                   display_labels = labels,
                                   cmap = plt.cm.Blues,
                                   xticks_rotation = 70,
                                   normalize = 'true')
    matrix.ax_.set_title(f'{set_name} set confustion matrix, with normalization')

    plt.show()

class LemmaTokenizer:
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in doc]

def remove_stopwords(doc):

    # Removes the stopwords from an input document.
    stop_words = stopwords.words('english')
    return [token for token in doc if ((token not in stop_words) and (token.lower() not in stop_words))]

def lowercase_tokens(doc):

    # Turns all letters provided in an input to lowercase.
    return [token.lower() for token in doc]

def lowercase_and_remove_stopwords(doc):

    # Removes stopwords and lowercase tokens from provided input data.
    stop_words = stopwords.words('english')
    return [token.lower() for token in doc if token.lower() not in stop_words]

def lower_unless_all_caps(string_):

    # Make all words in the input string lowercase unless the word is uppercase.
    words = string_.split()
    processed_words = [w.lower() if not (w.isupper() and len(w) > 1) else w for w in words]
    return ' '.join(processed_words)

def remove_single_characters(word_list, exception_list):

    # Removes all the single characters, except for those on the exempted list.
    return [w for w in word_list if (len(w) > 1 or w in exception_list)]

def remove_words(word_list, words_to_remove):

    # Remove all the words in the 'words_to_remove list' from the 'words_list'.
    return [w for w in word_list if w not in words_to_remove]

def tokenize_and_normalize_title_and_text(title, text):

    # Combines, tokenizes & normalizes the title and text of a news story.
    URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    TWITTER_HANDLE_REGEX = r'(?<=^|(?<=[^\w]))(@\w{1,15})\b'
    DATE_WORDS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
              'saturday', 'sunday', 'january', 'february', 'march', 'april',
             'may', 'june', 'july', 'august', 'september', 'october',
             'november', 'december']

    title_text = ' '.join([title, text])
    title_text = re.sub(URL_REGEX, '{link}', title_text)
    title_text = re.sub(TWITTER_HANDLE_REGEX, '@twitter-handle', title_text)
    title_text = lower_unless_all_caps(title_text)
    title_text = re.sub(r'\d+', ' ', title_text)
    title_text = re.sub(r'\(reuters\)', ' ', title_text)
    tokens = word_tokenize(title_text)
    tokens = remove_single_characters(tokens, ['i', '!'])
    tokens = remove_words(tokens, ["'s"])
    tokens = remove_words(tokens, DATE_WORDS)

    return tokens