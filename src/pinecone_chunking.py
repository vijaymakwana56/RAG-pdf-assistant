from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataChunking:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def chunk_documents(self, documents:List[Any])->List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n","\n"," ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Splitted {len(documents)} documents to {len(chunks)} chunks")
        return chunks