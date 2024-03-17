#KimiaEsmaili-610398193-MiniProject4

#1.document processing:
import os
import re
from bs4 import BeautifulSoup

def preprocess_documents(dataset_path):
    documents = []
    documents_text = []
    
    # Read all files in the dataset folder
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        
        # Read the content of each file
        with open(filepath, 'r') as file:
            content = file.read()
            
        # Use BeautifulSoup to extract text from TITLE and TEXT tags
        soup = BeautifulSoup(content, 'html.parser')
        title = soup.find('title').text.strip()
        text = soup.find('text').text.strip()
        
        # Preprocess the text (e.g., remove punctuation, convert to lowercase)
        title = preprocess_text(title)
        text = preprocess_text(text)
        
        # Store the document text
        documents_text.append(title + " " + text)
        
        # Store the document details (e.g., title, text)
        documents.append({
            'title': title,
            'text': text
        })
    
    return documents, documents_text

def preprocess_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Set the path to the Cranfield dataset folder
dataset_path = '/path/to/cranfield_dataset'

# Read and preprocess the documents
documents, documents_text = preprocess_documents(dataset_path)

# Print the preprocessed document text
for doc_text in documents_text:
    print(doc_text)
#----------------------------------------------------------------------------------

#2.Document Ranking – Space Vector Model:
# Import the necessary libraries:
import nltk
import math

# Define a function to calculate the
# term frequency (TF) for each term in a document
def calculate_tf(document):
    tf = {}
    tokens = nltk.word_tokenize(document)
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return tf

# Define a function to calculate the
# document frequency (DF) for each term in a collection of documents
def calculate_df(documents):
    df = {}
    for document in documents:
        tokens = set(nltk.word_tokenize(document))
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    return df

# Define a function to calculate the
# tf-idf weight for each term in a document.
def calculate_tfidf(tf, df, num_documents):
    tfidf = {}
    for term in tf:
        tfidf[term] = (1 + math.log(tf[term])) * math.log(num_documents / df[term])
    return tfidf

# Define the main function to perform document ranking.
def document_ranking(query, num_top_documents, documents):
    # Step 1: Calculate term frequency for the query
    query_tf = calculate_tf(query)

    # Step 2: Calculate document frequency for the collection of documents
    df = calculate_df(documents)

    # Step 3: Calculate tf-idf weights for the query
    query_tfidf = calculate_tfidf(query_tf, df, len(documents))

    # Step 4: Calculate tf-idf weights for each document in the collection
    documents_tfidf = []
    for document in documents:
        tf = calculate_tf(document)
        tfidf = calculate_tfidf(tf, df, len(documents))
        documents_tfidf.append(tfidf)

    # Step 5: Rank the documents based on their similarity to the query
    ranked_documents = []
    for i, document_tfidf in enumerate(documents_tfidf):
        similarity = 0
        for term in query_tfidf:
            if term in document_tfidf:
                similarity += query_tfidf[term] * document_tfidf[term]
        ranked_documents.append((i, similarity))

    # Step 6: Sort the ranked documents in descending order of similarity
    ranked_documents.sort(key=lambda x: x[1], reverse=True)

    # Step 7: Retrieve the top documents
    top_documents = []
    for i in range(min(num_top_documents, len(ranked_documents))):
        top_documents.append(documents[ranked_documents[i][0]])

    return top_documents
#----------------------------------------------------------------------------------

#3.Max-Heap – Space Vector Model:
import heapq
from nltk import tokenize

# Function to calculate the dot product of two vectors
def dot_product(vector1, vector2):
    return sum(i * j for i, j in zip(vector1, vector2))

# Function to calculate the magnitude of a vector
def vector_magnitude(vector):
    return sum(i ** 2 for i in vector) ** 0.5

# Function to calculate the cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    return dot_product(vector1, vector2) / (vector_magnitude(vector1) * vector_magnitude(vector2))

# Function to calculate the page rank using the Space Vector Model with Max-Heap
def page_rank_max_heap(documents, query, k):
    # Create a max-heap to store the top-k documents
    max_heap = []
    
    # Tokenize the query
    query_tokens = tokenize.word_tokenize(query)
    
    for document in documents:
        # Tokenize the document
        document_tokens = tokenize.word_tokenize(document)
        
        # Calculate the cosine similarity between the query and document vectors
        similarity = cosine_similarity(query_tokens, document_tokens)
        
        # Push the document and its similarity score into the max-heap
        heapq.heappush(max_heap, (similarity, document))

        # If the max-heap exceeds size k, remove the smallest element
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    
    # Get the top-k documents from the max-heap
    top_k_documents = [document for _, document in heapq.nlargest(k, max_heap)]
    
    return top_k_documents

# Example usage
documents = ["This is the first document.", "This document is the second document.", "And this is the third one."]
query = "This document is the first document."
k = 2

top_documents = page_rank_max_heap(documents, query, k)
print(top_documents)
#----------------------------------------------------------------------------------

#4.Document Ranking – Probabilistic Model:
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Assuming you have already defined the necessary functions for preprocessing and indexing the documents

def calculate_document_score(query, document, index, document_length, avg_document_length, total_documents):
    score = 0
    query_terms = word_tokenize(query.lower())
    document_terms = word_tokenize(document.lower())
    stop_words = set(stopwords.words('english'))
    
    for term in query_terms:
        if term not in stop_words:
            term_frequency = document_terms.count(term)
            document_frequency = index.get(term, 0)
            
            idf = math.log((total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            term_score = ((term_frequency * (1.2 + 1)) / (term_frequency + 1.2 * (1 - 0.75 + 0.75 * (document_length / avg_document_length)))) * idf
            
            score += term_score
    
    return score

def rank_documents(query, top_n):
    # Assuming you have already defined the necessary variables and data structures for indexing and storing documents
    
    # Calculate average document length
    total_words = sum(document_length.values())
    avg_document_length = total_words / len(document_length)
    
    # Create a list to store document scores
    document_scores = []
    
    # Iterate over all documents and calculate their scores
    for document_id, document in documents.items():
        score = calculate_document_score(query, document, index, document_length[document_id], avg_document_length, len(documents))
        document_scores.append((document_id, score))
    
    # Sort the document scores in descending order
    document_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve the top N documents
    top_documents = document_scores[:top_n]
    
    # Return the top documents
    return top_documents

# Call the rank_documents function with a query and number of top documents to retrieve
query = "example query"
top_n = 10
top_documents = rank_documents(query, top_n)
print(top_documents)
#----------------------------------------------------------------------------------

#5.Long queries – Probabilistic Model:
import math
from nltk.tokenize import word_tokenize

# Function to calculate Okapi BM25 score for a given query and document
def calculate_bm25(query, document, k1=1.2, b=0.75):
    # Tokenize query and document
    query_tokens = word_tokenize(query.lower())
    doc_tokens = word_tokenize(document.lower())
    
    # Calculate term frequencies
    query_term_freq = {term: query_tokens.count(term) for term in query_tokens}
    doc_term_freq = {term: doc_tokens.count(term) for term in doc_tokens}
    
    # Calculate document length
    doc_length = len(doc_tokens)
    
    # Calculate average document length
    avg_doc_length = sum(len(doc) for doc in documents) / len(documents)
    
    # Calculate Okapi BM25 score
    score = 0
    for term in query_tokens:
        if term in doc_tokens:
            idf = math.log(len(documents) / df[term])
            numerator = (doc_term_freq[term] * (k1 + 1))
            denominator = (doc_term_freq[term] + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            score += idf * (numerator / denominator)
    
    return score

# Function to handle long queries using Okapi BM25
def handle_long_queries(query, documents):
    # Define threshold for long queries
    threshold = 10
    
    # Check query length
    query_length = len(word_tokenize(query.lower()))
    
    if query_length > threshold:
        # Use Okapi BM25 approach
        scores = []
        for document in documents:
            score = calculate_bm25(query, document)
            scores.append(score)
        
        # Sort documents based on scores
        sorted_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
        
        return sorted_docs
    else:
        # Use previous function for short queries
        return search_documents(query, documents)

# Example usage
documents = ["This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"]
query = "This is the first document."

# Call the function to handle long queries
results = handle_long_queries(query, documents)
print(results)
#----------------------------------------------------------------------------------

#6.Document Ranking – Language Model:
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math

# Load the documents and create a inverted index
def create_inverted_index(documents):
    inverted_index = {}
    for doc_id, doc_text in documents.items():
        tokens = preprocess_text(doc_text)
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)
    return inverted_index

# Preprocess the text: tokenization, stopword removal, and stemming
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Calculate the term frequency-inverse document frequency (TF-IDF) score for a term in a document
def calculate_tfidf(term, doc_id, inverted_index, documents):
    tf = inverted_index[term].count(doc_id)
    idf = math.log(len(documents) / len(inverted_index[term]))
    return tf * idf

# Retrieve top documents based on TF-IDF scores for a query
def retrieve_top_documents(query, num_docs, inverted_index, documents):
    scores = {}
    for term in query:
        if term in inverted_index:
            for doc_id in inverted_index[term]:
                score = calculate_tfidf(term, doc_id, inverted_index, documents)
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += score
    
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_docs]
    return top_docs

# Language Model Document Ranking using Dirichlet Smoothing
def rank_documents_dirichlet(query, num_docs, inverted_index, documents):
    mu = 2000  # Parameter for Dirichlet Smoothing
    doc_lengths = {doc_id: len(preprocess_text(doc_text)) for doc_id, doc_text in documents.items()}
    collection_length = sum(doc_lengths.values())
    
    scores = {}
    for doc_id, doc_text in documents.items():
        tokens = preprocess_text(doc_text)
        doc_length = doc_lengths[doc_id]
        
        for term in query:
            # Calculate the term's probability in the document using Dirichlet Smoothing
            term_prob = (len(inverted_index[term]) / collection_length) if term in inverted_index else 0
            doc_term_freq = tokens.count(term)
            doc_prob = (doc_term_freq + mu * term_prob) / (doc_length + mu)
            
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += math.log(doc_prob)
    
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_docs]
    return top_docs

# Language Model Document Ranking using Jelinek-Mercer Smoothing
def rank_documents_jelinek(query, num_docs, inverted_index, documents):
    lambda_val = 0.7  # Parameter for Jelinek-Mercer Smoothing
    doc_lengths = {doc_id: len(preprocess_text(doc_text)) for doc_id, doc_text in documents.items()}
    collection_length = sum(doc_lengths.values())
    
    scores = {}
    for doc_id, doc_text in documents.items():
        tokens = preprocess_text(doc_text)
        doc_length = doc_lengths[doc_id]
        
        for term in query:
            # Calculate the term's probability in the document using Jelinek-Mercer Smoothing
            term_prob = (len(inverted_index[term]) / collection_length) if term in inverted_index else 0
            doc_term_freq = tokens.count(term)
            doc_prob = (lambda_val * doc_term_freq / doc_length) + ((1 - lambda_val) * term_prob)
            
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += math.log(doc_prob)
    
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_docs]
    return top_docs

# Main function for document ranking utilizing the language model
def document_ranking(query_text, num_docs, use_dirichlet_smoothing=True):
    # Load the documents
    documents = {
        'doc1': 'This is the first document',
        'doc2': 'This document talks about Python programming',
        'doc3': 'Python is a popular programming language',
        'doc4': 'Python has many useful libraries'
    }
    
    # Create the inverted index
    inverted_index = create_inverted_index(documents)
    
    # Preprocess the query text
    query = preprocess_text(query_text)
    
    # Rank the documents based on the chosen smoothing method
    if use_dirichlet_smoothing:
        top_docs = rank_documents_dirichlet(query, num_docs, inverted_index, documents)
        print(f"Using Dirichlet Smoothing:")
    else:
        top_docs = rank_documents_jelinek(query, num_docs, inverted_index, documents)
        print(f"Using Jelinek-Mercer Smoothing:")
    
    # Print the top ranked documents
    for doc_id, score in top_docs:
        print(f"Document ID: {doc_id}, Score: {score}")
        print(documents[doc_id])
        print()
#----------------------------------------------------------------------------------

#Evaluation function
def evaluate_model(model_results, relevant_documents):
    # Assuming model_results and relevant_documents are lists of document IDs
    true_positives = len(set(model_results) & set(relevant_documents))
    false_positives = len(set(model_results) - set(relevant_documents))
    false_negatives = len(set(relevant_documents) - set(model_results))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    accuracy = true_positives / len(relevant_documents) if len(relevant_documents) != 0 else 0

    return precision, recall, accuracy
