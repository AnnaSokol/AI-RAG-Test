#%% packages
import os
from pprint import pprint
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# %% set up database connection
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
parent_dir = os.path.dirname(current_dir)

chroma_dir = os.path.join(parent_dir, "db")

# %% embedding function (MUSS identisch zum Index sein!)
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# %% connect to the database
db = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embedding_function
)

# %% retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

# %% queries (einzeln testen)
# query = "Who is the sidekick of Sherlock Holmes in the book?"
# query = "Find passages that describe the moor or its atmosphere."
# query = "Which chapters or passages convey a sense of fear or suspense?"
# query = "Identify all conversations between Sherlock Holmes and Dr. Watson."
query = "How does the hound look like?"

# %% retrieval
most_similar_docs = retriever.invoke(query)

# %% inspect result
print(f"Retrieved documents: {len(most_similar_docs)}\n")
pprint(most_similar_docs[0].page_content)

