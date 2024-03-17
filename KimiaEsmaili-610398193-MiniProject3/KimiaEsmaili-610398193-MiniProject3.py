import re
import heapq
import math
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


    # Document Preprocessing
#def preprocess_document(doc):
    # Tokenization and lowercasing
#    tokens = re.findall(r'\b\w+\b', doc.lower())
    # Stemming
    # (You can use libraries like NLTK or SpaCy for stemming)
    # Punctuation removal
#    tokens = [token for token in tokens if token.isalnum()]
#    return tokens

# BSBI Algorithm
class BSBIIndex:
    def __init__(self):
        self.term_id_map = {}
        self.term_posting_lists = {}
        self.current_doc_id = 0

    def add_doc(self, doc):
        terms = preprocess_document(doc)
        doc_id = self.current_doc_id
        self.current_doc_id += 1
        for term in set(terms):
            term_id = self.term_id_map.setdefault(term, len(self.term_id_map))
            posting_list = self.term_posting_lists.setdefault(term_id, [])
            # Add document ID to the posting list
            posting_list.append(doc_id)

    def gamma_encode(self, num):
        # Function to perform gamma encoding of the document ID gaps
        binary_num = bin(num)[2:]
        offset = binary_num[1:]
        length = '1' * len(offset) + '0'
        return length + offset

    def build_index(self):
        # Sort the posting lists by document ID
        for term_id, posting_list in self.term_posting_lists.items():
            self.term_posting_lists[term_id] = sorted(posting_list)

    def merge_blocks(self):
	# Assuming that we have multiple intermediate index blocks stored in separate files
        intermediate_blocks = ['block1.txt', 'block2.txt', 'block3.txt']  # Example file names

        merged_index = {}  # Merged index will be stored in memory

        # Open each intermediate block and merge the posting lists
        for block_file in intermediate_blocks:
            with open(block_file, 'r') as file:
                for line in file:
                    term_id, posting_list = line.strip().split('\t')
                    term_id = int(term_id)
                    posting_list = [int(doc_id) for doc_id in posting_list.split(',')]
                    if term_id in merged_index:
                        # Merge posting lists for the same term
                        merged_index[term_id].extend(posting_list)
                    else:
                        merged_index[term_id] = posting_list

        # Sort the merged posting lists by document ID
        for term_id, posting_list in merged_index.items():
            merged_index[term_id] = sorted(posting_list)

        # Save the merged index to a new file
        with open('merged_index.txt', 'w') as merged_file:
            for term_id, posting_list in merged_index.items():
                merged_file.write(f"{term_id}\t{','.join(str(doc_id) for doc_id in posting_list)}\n")

# Example usage
index = BSBIIndex()
docs = ["The quick brown fox jumps over the lazy dog",
        "An apple a day keeps the doctor away",
        "The early bird catches the worm"]
for doc in docs:
    index.add_doc(doc)

index.build_index()
index.merge_blocks()
