#%% packages
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#%% Vector store path
persist_directory = "rag_store"

#%% Create or load vector store
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(persist_directory):
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
else:
    data = WikipediaLoader(
        query="Human history",
        load_max_docs=20,
        doc_content_chars_max=5000,
    ).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(data)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

#%% Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

#%% Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}"
    ),
    ("human", "{question}")
])

#%% Local LLM
model = ChatOllama(
    model="gemma3:1b",   # klein & lokal
    temperature=0
)

chain = prompt | model | StrOutputParser()

#%% Simple RAG function
def simple_rag_system(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return chain.invoke({"question": question, "context": context})

#%% Test
question = "What happened in the First World War?"
print(simple_rag_system(question))

