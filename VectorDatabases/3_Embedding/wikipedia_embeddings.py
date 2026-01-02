#%% Packages
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pprint import pprint

# %% Load the article
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(
    query=ai_article_title,
    load_all_available_meta=True,
    doc_content_chars_max=10000,
    load_max_docs=1
)
doc = loader.load()

# %% Create splitter instance
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ".", ","]
)

# %% Apply chunking
chunks = splitter.split_documents(doc)
print(f"Number of chunks: {len(chunks)}")

# %% Local embedding model (NO OpenAI)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# %% extract texts
texts = [chunk.page_content for chunk in chunks]

# %% create embeddings
embeddings = embeddings_model.embed_documents(texts)

# %% inspect embeddings
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")

