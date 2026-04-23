import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR    = os.path.join(BASE_DIR, "Data", "Docs")
VECTORS_DIR = os.path.join(BASE_DIR, "Data", "Vectors")

print(DOCS_DIR)
print(VECTORS_DIR)

def get_embedings():
    embedding = HuggingFaceEmbeddings()
    return embedding

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key = st.secrets["GEMINI_API_KEY"],
        temperature = 0.1
    )
    return llm

def process_document_to_chroma_db():
    loader = DirectoryLoader(
        path = DOCS_DIR ,
        glob = "./*.pdf",
        loader_cls = UnstructuredFileLoader
    )

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =2000,
        chunk_overlap = 200
    )

    texts = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=get_embedings(),
        persist_directory=VECTORS_DIR
    )

    return 0


def process_answer(question):
    vectordb = Chroma(
        persist_directory=VECTORS_DIR,
        embedding_function=get_embedings()
    )

    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type = "stuff",
        retriever = retriever
    )

    response = qa_chain.invoke({"query": question})
    answer = response["result"]

    return answer


