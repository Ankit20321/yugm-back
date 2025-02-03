from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    """ Preprocess text by lowercasing, removing stopwords, and stemming """
    tokens = text.lower().split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return tokens

def expand_query(query):
    """ Expand the query using synonyms from WordNet """
    synonyms = set(query.split())
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return ' '.join(synonyms)

def get_embeddings(texts):
    """ Get embeddings using Sentence Transformer """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def compute_tfidf(corpus, query):
    """ Compute TF-IDF scores for the corpus """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return cosine_sim

def rank_chunks_with_bm25(chunks, query):
    """
    Rank chunks first by cosine similarity (Sentence Transformer embeddings), rerank using BM25,
    and combine with TF-IDF for final ranking.
    Args:
        chunks (list): List of document chunks (objects with 'page_content' attribute).
        query (str): Query string for which to rank the chunks.
    Returns:
        List of (chunk, score) sorted by combined relevance.
    """
    # Step 1: Preprocess chunk texts and query
    chunk_texts = [chunk.page_content for chunk in chunks]
    preprocessed_chunks = [' '.join(preprocess_text(chunk.page_content)) for chunk in chunks]
    preprocessed_query = ' '.join(preprocess_text(query))
    
    # Step 2: Expand the query using WordNet
    expanded_query = expand_query(preprocessed_query)
    
    # Step 3: Get embeddings for chunks and query for semantic similarity
    chunk_embeddings = get_embeddings(chunk_texts)
    query_embedding = get_embeddings([expanded_query])
    
    # Step 4: Calculate cosine similarity between query and chunks
    similarity_scores = cosine_similarity(query_embedding.cpu(), chunk_embeddings.cpu())[0]
    
    # Step 5: Rank chunks by cosine similarity score
    ranked_chunks = [(chunk, score) for chunk, score in zip(chunks, similarity_scores)]
    ranked_chunks = sorted(ranked_chunks, key=lambda x: x[1], reverse=True)

    # Step 6: Rerank using BM25
    bm25_corpus = [preprocess_text(chunk.page_content) for chunk, _ in ranked_chunks]  # Tokenize chunks
    bm25 = BM25Okapi(bm25_corpus, k1=1.5, b=0.75)  # Tune BM25 parameters
    query_tokens = preprocess_text(expanded_query)
    
    # Get BM25 scores
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Step 7: Compute TF-IDF scores for additional reranking
    tfidf_scores = compute_tfidf(preprocessed_chunks, expanded_query)
    
    # Step 8: Combine BM25 scores with TF-IDF scores
    combined_scores = 0.5 * np.array(bm25_scores) + 0.5 * np.array(tfidf_scores)
    
    # Step 9: Combine and sort chunks by combined score
    final_reranked_chunks = [(chunk, combined_score) for (chunk, _), combined_score in zip(ranked_chunks, combined_scores)]
    final_reranked_chunks = sorted(final_reranked_chunks, key=lambda x: x[1], reverse=True)

    return final_reranked_chunks

# Example usage:
# Assuming you have loaded the chunks
# directory_path = '../data/Organizational Information/'
#chunks = load_n_split(path=directory_path, splitter="token")

# Example query
# query = "I want to know if there are any internal openings for me?"

# Rank the chunks based on the query, and then rerank using BM25 and TF-IDF
# reranked_chunks = rank_chunks_with_bm25_and_tfidf(chunks, query)

# Print the ranked chunks and their final scores
# for i, (chunk, score) in enumerate(reranked_chunks):
#     print(f"Rank {i+1} (Combined Score: {score}):\n{chunk.page_content}\n")
