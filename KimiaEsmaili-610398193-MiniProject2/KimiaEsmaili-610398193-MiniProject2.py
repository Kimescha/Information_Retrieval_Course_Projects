#Kimia Esmaili-610398193-MiniProject2
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.metrics.distance import edit_distance

# Step 1: Reading and Preprocessing Text Documents
def preprocess_documents(documents):
    processed_docs = []
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for doc in documents:
        # Remove special characters and punctuation
        doc = re.sub(r'[^\w\s]', '', doc)
        # Convert to lowercase
        doc = doc.lower()
        # Tokenize the document
        words = doc.split()
        # Remove stop words
        words = [word for word in words if word not in stop_words]
        # Perform stemming
        words = [stemmer.stem(word) for word in words]
        # Add the preprocessed document to the list
        processed_docs.extend(words)

    return processed_docs

# Step 2: Building an Inverted Index
def build_inverted_index(processed_docs):
    inverted_index = {}

    for idx, word in enumerate(processed_docs):
        if word not in inverted_index:
            inverted_index[word] = []
        inverted_index[word].append(idx)

    return inverted_index

# Step 3: Spelling Correction
def spelling_correction(query, word_list):
    query = query.lower()
    words = word_list
    closest_word = None
    min_distance = float('inf')

    for word in words:
        distance = edit_distance(query, word)
        if distance < min_distance:
            min_distance = distance
            closest_word = word

    return closest_word

# Step 4: Wildcard Queries
def wildcard_query(query, inverted_index):
    result = set()
    query = query.lower()
    query = query.translate(str.maketrans('', '', string.punctuation))
    query_terms = query.split()

    for term in query_terms:
        if '*' in term:
            prefix, suffix = term.split('*')
            for key in inverted_index.keys():
                if key.startswith(prefix) and key.endswith(suffix):
                    result.update(inverted_index[key])
        else:
            if term in inverted_index:
                result.update(inverted_index[term])

    return list(result)

# Example usage
documents = [
    "This is a simple example document. It contains several words. The words should be processed.",
    "Another example document with different content. Spelling correction is important for retrieval.",
    "Another example document to test Boolean search capabilities. This document contains relevant content.",
    "Is this the first document"
]

# Step 1: Preprocess Documents
processed_documents = preprocess_documents(documents)

# Step 2: Build Inverted Index
inverted_index = build_inverted_index(processed_documents)

# Step 3: Spelling Correction
query = "documnt"
closest_word = spelling_correction(query, processed_documents)
print("Spelling Correction Result:", closest_word)

# Step 4: Wildcard Queries
wildcard_query = "th*"
result_wildcard = wildcard_query(wildcard_query, inverted_index)
print("Wildcard Query Result:", result_wildcard)