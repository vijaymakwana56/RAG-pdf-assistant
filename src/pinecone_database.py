from pinecone import Pinecone, ServerlessSpec
from typing import List, Any
import os
import time
from src.embedding import EmbeddingPipeline
from src.pinecone_chunking import DataChunking
from src.data_loader import load_all_documents
from dotenv import load_dotenv

load_dotenv()

class PineconeVectorstore:
    def __init__(self, index_name:str="pdf-assistant", namespace: str="pdf_doc"):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(self.api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

    def upsert_chunks(self,userID: str, document_name: str, chunks: List[Any]):
        # This function adds document chunks into pinecone store
        records = []
        batch_size = 90 #to insert into pinecone database in batched
        for i, chunk in enumerate(chunks):
            records.append({
                "_id": f"chunk-{i}",
                "chunk_text": chunk.page_content,
                "document_name": document_name
            })
        
        for i in range(0,len(records),batch_size):
            batch = records[i:i+batch_size]
            self.index.upsert_records(namespace=userID, records=batch)
            time.sleep(3)
            
