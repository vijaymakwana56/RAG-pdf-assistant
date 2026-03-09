import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pinecone import Pinecone

load_dotenv()

class PRAGSearch:
    def __init__(self, index_name="pdf-assistant", llm_model='llama-3.3-70b-versatile'):
        #Load or build model
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(self.api_key)
        self.index = self.pc.Index(index_name)
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=groq_api_key,model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, userID:str, query:str, top_k:int=5)->str:
        results = self.index.search(
            namespace=userID,
            query={
                "inputs": {"text": f"{query}"},
                "top_k": top_k
            }
        )
        text = [r.fields["chunk_text"] for r in results.result.hits]
        context = '\n\n'.join(text)
        if not context:
            return "No relevent documents found."
        prompt = f"""You are a pdf document assistant.
        Summarize the following context for the query: {query} \n\nContext: {context}"""
        response = self.llm.invoke(prompt)
        return response.content