from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline


if __name__ == "__main__": 
    docs = load_all_documents('pdfs')
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunk_vectors = EmbeddingPipeline().embed_chunks(chunks)

    print(len(chunk_vectors))
    print(chunk_vectors[0])