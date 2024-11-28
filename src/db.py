from langchain_chroma.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import dotenv
import os
from typing_extensions import List

dotenv.load_dotenv()

class Database():
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001',google_api_key=self.api_key) 
        self.db_path = 'data/processed'

    def persist(self,docs:List[Document]):
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.db_path
            )
        return vectorstore
    
    def get_retriever(self):
        return Chroma(persist_directory=self.db_path).as_retriever()