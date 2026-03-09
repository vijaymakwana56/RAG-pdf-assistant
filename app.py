from src.data_loader import load_all_documents
# from src.search import RAGSearch
from src.search_on_pinecone import PRAGSearch


if __name__ == "__main__": 
    # docs = load_all_documents('pdfs')
    # store = FaissVectorStore("faiss_store")
    # # store.build_from_documents(docs)
    # store.load()
    # print(store.query("what is encoder-decoder architecture?",top_k=3))
    query = "what is encoder-decoder architecture?"
    # rag_search = RAGSearch()
    rag_search = PRAGSearch()
    summary = rag_search.search_and_summarize(userID="user1",query=query,top_k=3)
    print(f"summary: {summary}")