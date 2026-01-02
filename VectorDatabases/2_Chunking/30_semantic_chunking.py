#%% Packages
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WikipediaLoader
from langchain_ollama import OllamaEmbeddings
from pprint import pprint

# %% Load the article
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(
    query=ai_article_title,
    load_all_available_meta=True,
    doc_content_chars_max=1000,
    load_max_docs=1
)
doc = loader.load()

# %% check the content
print(doc[0].page_content[:500])

# %% Create splitter instance (LOCAL embeddings)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="cosine",
    breakpoint_threshold=0.5
)

# %% Apply semantic chunking
chunks = splitter.split_documents(doc)

# %% check the results
print(f"Number of chunks: {len(chunks)}\n")

pprint(chunks[0].page_content[:500])
if len(chunks) > 1:
    pprint(chunks[1].page_content[:500])

