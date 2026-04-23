import os ,sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Backend.RAG_Ingestion import process_document_to_chroma_db, process_answer

st.title("RAGbot")

uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file is not None:
    docs_path = os.path.join(os.path.dirname(__file__), "..", "Data", "Docs")

    os.makedirs(docs_path, exist_ok=True)

    save_path = os.path.join(docs_path,uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    process_document = process_document_to_chroma_db()
    st.success(f"File saved to {save_path}")

user_question = st.text_area("Ask your Question about the Document")


if st.button("Answer"):
    answer = process_answer(user_question)
    st.markdown("Response")
    st.markdown(answer)

