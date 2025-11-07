import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str='faiss_store', model_name: str='all-MiniLM-L6-v2', chunk_size=1000, chunk_overlap=200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir,exist_ok=True)
        self.Index = None
        self.metadata = []
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loading embedding model {self.model_name}")

    def add_embeddings(self, embeddings:np.ndarray, metadatas:List[Any]):
        dim = embeddings.shape[1]
        if self.Index is None:
            self.Index = faiss.IndexFlatL2(dim)
            self.Index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS Index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir,"metadata.pkl")
        faiss.write_index(self.Index,faiss_path)
        with open(meta_path,'wb') as f:
            pickle.dump(self.metadata,f)
        print(f"[INFO] Saved FAISS Index and metadata to {self.persist_dir}")

    def build_from_documents(self, documents:List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        embed_pipe = EmbeddingPipeline(model_name=self.model_name, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = embed_pipe.chunk_documents(documents)
        embeddings = embed_pipe.embed_chunks(chunks)
        metadata = [{'text': chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'),metadata)
        self.save()

    def load(self):
        faiss_path = os.path.join(self.persist_dir,"faiss.index")
        meta_path = os.path.join(self.persist_dir,"metadata.pkl")
        self.Index = faiss.read_index(faiss_path)
        with open(meta_path,'rb') as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self,query_embedding:np.ndarray, top_k:int=5):
        D, I = self.Index.search(query_embedding,top_k)
        results = []
        for idx, dist in zip(I[0],D[0]):
            meta = self.metadata[idx] if idx<len(self.metadata) else None
            results.append({'index':idx, 'distance':dist, 'metadata': meta})
        return results
    
    def query(self, query_text:str,top_k = 5):
        print(f"[INFO] Querying the Vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb,top_k=top_k)