#Kimia Esmaili-610398193-MiniProject1
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
        processed_docs.append(words)

    return processed_docs

# Step 2: Creating an Inverted Index
def create_inverted_index(processed_docs):
    inverted_index = {}

    for idx, doc in enumerate(processed_docs):
        for word in doc:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(idx)

    return inverted_index

# Step 3: Handling Standard Boolean Queries
def standard_boolean_query(query, inverted_index):
    query_terms = re.findall(r'\w+', query.lower())
    operators = re.findall(r'AND|OR|NOT', query.upper())
    result = inverted_index[query_terms[0]]

    for i in range(len(operators)):
        if operators[i] == 'AND':
            result = set(result) & set(inverted_index[query_terms[i+1]])
        elif operators[i] == 'OR':
            result = set(result) | set(inverted_index[query_terms[i+1]])
        elif operators[i] == 'NOT':
            result = set(result) - set(inverted_index[query_terms[i+1]])

    return list(result)

# Step 4: Handling Proximity Queries
def proximity_query(query, inverted_index, max_distance):
    query_terms = re.findall(r'\w+', query.lower())
    result = []

    for doc_id in inverted_index[query_terms[0]]:
        for term in query_terms[1:]:
            if any(abs(doc_id - x) <= max_distance for x in inverted_index[term]):
                result.append(doc_id)
                break

    return result

# Example usage
documents = [
    "This is a simple example document. It contains several words. The words should be processed and indexed.",
    "Another example document with different content. Document indexing is important for retrieval.",
    "Another example document to test Boolean search capabilities. This document contains relevant content.",
    "Is this the first document"
]

# Step 1: Preprocess Documents
processed_documents = preprocess_documents(documents)

# Step 2: Create Inverted Index
inverted_index = create_inverted_index(processed_documents)

# Step 3: Handle Standard Boolean Query
boolean_query = "document AND NOT second"
result_boolean = standard_boolean_query(boolean_query, inverted_index)
print("Standard Boolean Query Result:", result_boolean)

# Step 4: Handle Proximity Query
proximity_query = "first document"
max_distance = 2
result_proximity = proximity_query(proximity_query, inverted_index, max_distance)
print("Proximity Query Result:", result_proximity)