#%% packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from langchain_ollama import OllamaEmbeddings

#%% Documents
docs = [
    "The weather tomorrow will be sunny with a slight chance of rain.",
    "Dogs are known to be loyal and friendly companions to humans.",
    "The climate in tropical regions is warm and humid, often with frequent rain.",
    "Python is a powerful programming language used for machine learning.",
    "The temperature in deserts can vary widely between day and night.",
    "Cats are independent animals, often more solitary than dogs.",
    "Artificial intelligence and machine learning are rapidly evolving fields.",
    "Hiking in the mountains is an exhilarating experience, but it can be unpredictable due to weather changes.",
    "Winter sports like skiing and snowboarding require specific types of weather conditions.",
    "Programming languages like Python and JavaScript are popular choices for web development."
]

#%% Remove stopwords (Sparse Search)
docs_without_stopwords = [
    " ".join([w for w in doc.split() if w.lower() not in ENGLISH_STOP_WORDS])
    for doc in docs
]

#%% Sparse Search (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs_without_stopwords)

user_query = "Which weather is good for outdoor activities?"

query_sparse_vec = vectorizer.transform([user_query])
sparse_similarities = cosine_similarity(query_sparse_vec, tfidf_matrix).flatten()

#%% Helper: filter indices
def get_filtered_indices(similarities, threshold=0.0):
    return [
        i for i, sim in sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True
        )
        if sim > threshold
    ]

filtered_sparse = get_filtered_indices(sparse_similarities, threshold=0.2)

#%% Dense Search (LOCAL embeddings)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

embedded_docs = embeddings.embed_documents(docs)
query_dense_vec = embeddings.embed_query(user_query)

dense_similarities = cosine_similarity([query_dense_vec], embedded_docs)[0]
filtered_dense = get_filtered_indices(dense_similarities, threshold=0.8)

#%% Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(sparse, dense, alpha=0.3):
    scores = {}

    for rank, idx in enumerate(sparse, start=1):
        scores[idx] = scores.get(idx, 0) + alpha / (rank + 60)

    for rank, idx in enumerate(dense, start=1):
        scores[idx] = scores.get(idx, 0) + (1 - alpha) / (rank + 60)

    return sorted(scores, key=scores.get, reverse=True)

#%% Final ranking
final_indices = reciprocal_rank_fusion(filtered_sparse, filtered_dense)
print("Final ranked documents:\n")

for i in final_indices:
    print(f"- {docs[i]}")

