#%% Packages
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


#%% Create chunks from a text file
def create_chunks(text_file_name: str) -> list[Document]:
    # Path handling
    file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(current_dir)

    text_file_path = os.path.join(parent_dir, "data", text_file_name)

    # Load document
    loader = TextLoader(file_path=text_file_path, encoding="utf-8")
    docs = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ".", ","]
    )
    chunks = splitter.split_documents(docs)

    return chunks

