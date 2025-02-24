import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import nltk

# Ensure required NLTK resources are available.
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def passthrough(doc):
    """
    Passthrough function for use in the pipeline if the text is already tokenized.
    """
    return doc


def confusion_matrix_and_classification_report(estimator, X, y, labels, set_name):
    """
    Display the classification report and confusion matrices (both raw and normalized)
    for the given estimator on the dataset.
    """
    predictions = estimator.predict(X)
    print(f'Classification report for {set_name} set:')
    print(classification_report(y, predictions, target_names=labels))

    # Display confusion matrix without normalization.
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator,
        X,
        y,
        display_labels=labels,
        cmap=plt.cm.Blues,
        xticks_rotation=70,
        normalize=None
    )
    disp.ax_.set_title(f'{set_name} set confusion matrix, without normalization')
    plt.show()

    # Display confusion matrix with normalization.
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator,
        X,
        y,
        display_labels=labels,
        cmap=plt.cm.Blues,
        xticks_rotation=70,
        normalize='true'
    )
    disp.ax_.set_title(f'{set_name} set confusion matrix, with normalization')
    plt.show()


class LemmaTokenizer:
    """
    Tokenizer that lemmatizes tokens using NLTK's WordNetLemmatizer.
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in doc]


def remove_stopwords(doc):
    """
    Remove English stopwords from an input document.
    """
    stop_words = stopwords.words('english')
    return [token for token in doc if token.lower() not in stop_words]


def identity_function(x):
    return x


def lowercase_tokens(doc):
    """
    Convert all tokens in the input document to lowercase.
    """
    return [token.lower() for token in doc]


def lowercase_and_remove_stopwords(doc):
    """
    Lowercase tokens and remove English stopwords from the document.
    """
    stop_words = stopwords.words('english')
    return [token.lower() for token in doc if token.lower() not in stop_words]


def lower_unless_all_caps(string_):
    """
    Convert words in the string to lowercase unless the word is fully uppercase (and longer than one character).
    """
    words = string_.split()
    processed_words = [w.lower() if not (w.isupper() and len(w) > 1) else w for w in words]
    return ' '.join(processed_words)


def remove_single_characters(word_list, exception_list):
    """
    Remove single-character tokens from the list, except those specified in the exception_list.
    """
    return [w for w in word_list if (len(w) > 1 or w in exception_list)]


def remove_words(word_list, words_to_remove):
    """
    Remove all words in the 'words_to_remove' list from the word_list.
    """
    return [w for w in word_list if w not in words_to_remove]


def tokenize_and_normalize_title_and_text(title, text):
    """
    Combine, tokenize, and normalize the title and text of a news story.
    This function:
      - Replaces URLs and Twitter handles with placeholders.
      - Lowercases text unless the word is fully uppercase.
      - Removes digits and specific unwanted tokens.
      - Tokenizes the result and removes single-character tokens (with exceptions) and unwanted words.
    """
    URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    TWITTER_HANDLE_REGEX = r'(?<=^|(?<=[^\w]))(@\w{1,15})\b'
    DATE_WORDS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                  'saturday', 'sunday', 'january', 'february', 'march', 'april',
                  'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']

    # Combine title and text.
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
