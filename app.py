from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore


if __name__ == "__main__": 
    docs = load_all_documents('pdfs')
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    print(store.query("what is encoder-decoder architecture?",top_k=3))