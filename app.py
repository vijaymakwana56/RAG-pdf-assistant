import os
import streamlit as st
from src.data_loader import load_all_documents
from src.pinecone_chunking import DataChunking
from src.pinecone_database import PineconeVectorstore
from src.search_on_pinecone import PRAGSearch

st.title("Chat With Your PDFs")

upload_dir = "pdfs"
os.makedirs(upload_dir, exist_ok=True)

if "rag" not in st.session_state:
    st.session_state.rag = PRAGSearch()

ck = DataChunking()
db = PineconeVectorstore()

userId = "user2"

#-----------PDF Uploader----------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join(upload_dir, uploaded_file.name)

    with open(save_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Sucessfully")

    #load documents
    docs = load_all_documents(upload_dir)

    #creating chunks
    chunks = ck.chunk_documents(documents=docs)

    #sotre into pinecone database
    db.upsert_chunks(
        userID=userId,
        document_name=uploaded_file.name,
        chunks=chunks
    )

    st.success("Documents processed and stored.")


#----------Chat Section----------
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Ask a question about your document")

if prompt:
    st.session_state.messages.append({"role":"user", "content":prompt})

    answer = st.session_state.rag.search_and_summarize(userID=userId,query=prompt)

    st.session_state.messages.append({"role":"assistant", "content":answer})


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])