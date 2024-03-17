#KimiaEsmaili-610398193-MiniProject5
#1,2
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
#3,4
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
#5
from gensim.models import KeyedVectors
from sklearn.svm import SVC
#6
from sklearn.metrics import classification_report
####part 1 and 2:

# Load negative and positive train dataset files
train_neg = pd.read_csv('train_neg.csv')
train_pos = pd.read_csv('train_pos.csv')

# Load negative and positive test dataset files
test_neg = pd.read_csv('test_neg.csv')
test_pos = pd.read_csv('test_pos.csv')

# Combine negative and positive datasets for train and test
train_data = pd.concat([train_neg, train_pos])
test_data = pd.concat([test_neg, test_pos])

# Preprocess the training documents
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess the text data
def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    #text = ' '.join(words)
    return text

train_data['preprocessed_text'] = train_data['text'].apply(preprocess_text)
test_data['preprocessed_text'] = test_data['text'].apply(preprocess_text)

# Convert documents into a word list
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['preprocessed_text'])
X_test = vectorizer.fit_transform(test_data['preprocessed_text'])

# Naive Bayes Classification
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, train_data['label'])

# Evaluate the classifier on the training dataset
train_predictions = nb_classifier.predict(X_train)
train_accuracy = accuracy_score(train_data['label'], train_predictions)
print("Training Accuracy with Naive Bayes: ", train_accuracy)
########################################################################################
#part 3 and 4
# Word Embeddings with SVM Classification

# Apply Word2Vec for word embeddings
word2vec_model = Word2Vec(train_data['preprocessed_text'].apply(str.split), vector_size=100, window=5, min_count=1, sg=1)
word_vectors = word2vec_model.wv

# Transform word embeddings into document embeddings
def doc_embedding(doc):
    vector_sum = np.zeros(100)
    for word in doc.split():
        if word in word_vectors:
            vector_sum += word_vectors[word]
    return vector_sum

X_train_word2vec = np.array([doc_embedding(doc) for doc in train_data['preprocessed_text']])

# SVM Classification with Word Embeddings
svm_classifier_word2vec = SVC()
svm_classifier_word2vec.fit(X_train_word2vec, train_data['label'])

# Evaluate the classifier
train_predictions_word2vec = svm_classifier_word2vec.predict(X_train_word2vec)
train_accuracy_word2vec = accuracy_score(train_data['label'], train_predictions_word2vec)
print("Training Accuracy with Word2Vec: ", train_accuracy_word2vec)

# LSA with SVM Classification

# Apply Latent Semantic Analysis (LSA)
lsa = TruncatedSVD(n_components=100)
X_train_lsa = lsa.fit_transform(X_train)

# SVM Classification with LSA-transformed word embedding vectors
svm_classifier_lsa = SVC()
svm_classifier_lsa.fit(X_train_lsa, train_data['label'])

# Evaluate the classifier
train_predictions_lsa = svm_classifier_lsa.predict(X_train_lsa)
train_accuracy_lsa = accuracy_score(train_data['label'], train_predictions_lsa)
print("Training Accuracy with LSA: ", train_accuracy_lsa)
########################################################################################
#part 5


# Load pre-trained GloVe and FastText embeddings
glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False)
fasttext_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

# Transform word embeddings into document embeddings
def doc_embedding_glove(doc):
    vector_sum = np.zeros(100)
    for word in doc.split():
        if word in glove_model:
            vector_sum += glove_model[word]
    return vector_sum

def doc_embedding_fasttext(doc):
    vector_sum = np.zeros(300)
    for word in doc.split():
        if word in fasttext_model:
            vector_sum += fasttext_model[word]
    return vector_sum

X_train_glove = np.array([doc_embedding_glove(doc) for doc in train_data['preprocessed_text']])
X_train_fasttext = np.array([doc_embedding_fasttext(doc) for doc in train_data['preprocessed_text']])

X_test_glove = np.array([doc_embedding_glove(doc) for doc in test_data['preprocessed_text']])
X_test_fasttext = np.array([doc_embedding_fasttext(doc) for doc in test_data['preprocessed_text']])

# SVM Classification with GloVe
svm_classifier_glove = SVC()
svm_classifier_glove.fit(X_train_glove, train_data['label'])

# Evaluate the classifier
train_predictions_glove = svm_classifier_glove.predict(X_train_glove)
train_accuracy_glove = accuracy_score(train_data['label'], train_predictions_glove)
print("Training Accuracy with GloVe: ", train_accuracy_glove)

# SVM Classification with FastText
svm_classifier_fasttext = SVC()
svm_classifier_fasttext.fit(X_train_fasttext, train_data['label'])

# Evaluate the classifier
train_predictions_fasttext = svm_classifier_fasttext.predict(X_train_fasttext)
train_accuracy_fasttext = accuracy_score(train_data['label'], train_predictions_fasttext)
print("Training Accuracy with FastText: ", train_accuracy_fasttext)
########################################################################################
#part 6


# Transform test data into document embeddings for Word2Vec
X_test_word2vec = np.array([doc_embedding(doc) for doc in test_data['preprocessed_text']])

# Evaluate Naive Bayes classifier on the test dataset
test_predictions_nb = nb_classifier.predict(X_test)
test_accuracy_nb = accuracy_score(test_data['label'], test_predictions_nb)
print("Test Accuracy with Naive Bayes: ", test_accuracy_nb)

# Evaluate SVM classifiers on the test dataset for Word2Vec, GloVe, and FastText
test_predictions_word2vec = svm_classifier_word2vec.predict(X_test_word2vec)
test_accuracy_word2vec = accuracy_score(test_data['label'], test_predictions_word2vec)
print("Test Accuracy with Word2Vec: ", test_accuracy_word2vec)

test_predictions_glove = svm_classifier_glove.predict(X_test_glove)
test_accuracy_glove = accuracy_score(test_data['label'], test_predictions_glove)
print("Test Accuracy with GloVe: ", test_accuracy_glove)

test_predictions_fasttext = svm_classifier_fasttext.predict(X_test_fasttext)
test_accuracy_fasttext = accuracy_score(test_data['label'], test_predictions_fasttext)
print("Test Accuracy with FastText: ", test_accuracy_fasttext)

# Classification report for SVM classifiers
print("Classification Report for Word2Vec:")
print(classification_report(test_data['label'], test_predictions_word2vec))

print("Classification Report for GloVe:")
print(classification_report(test_data['label'], test_predictions_glove))

print("Classification Report for FastText:")
print(classification_report(test_data['label'], test_predictions_fasttext))

