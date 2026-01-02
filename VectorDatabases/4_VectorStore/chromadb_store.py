#%% Packages
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

#%% Path handling
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
parent_dir = os.path.dirname(current_dir)

text_file_path = os.path.join(parent_dir, "data", "HoundOfBaskerville.txt")
persistent_db_path = os.path.join(parent_dir, "db")

#%% Load document
loader = TextLoader(file_path=text_file_path, encoding="utf-8")
docs = loader.load()

#%% Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ".", ","]
)
chunks = splitter.split_documents(docs)
print(f"Number of chunks: {len(chunks)}")

#%% Local embeddings (Ollama)
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

#%% Create & persist Chroma DB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=persistent_db_path
)

print(f"Number of stored documents: {len(db.get()['ids'])}")
