from pathlib import Path
from typing import List,Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str)->List[Any]:
    #use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    #pdf files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] found {len(pdf_files)} pdf files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] loading file {pdf_file}")
        try:
            pdf_loader = PyMuPDFLoader(str(pdf_file))
            loaded = pdf_loader.load()
            print(f"[DEBUG] loaded {len(loaded)} PDF documents of {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load {pdf_file}: {e}")

    # text files

    #cvs files

    #sql files
    return documents