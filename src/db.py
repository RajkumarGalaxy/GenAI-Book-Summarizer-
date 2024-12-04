from langchain_chroma.vectorstores import Chroma 
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing_extensions import List
import os 

from utils import get_embedding

class Database():
    def __init__(self, raw_path, chunk_size = 500, chunk_overlap = 100):
        raw_path = os.path.splitext(raw_path)[0]
        raw_path = os.path.split(raw_path)[-1]
        self.db_path = 'data/processed/' + raw_path
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap

    def split(self, doc:Document):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(documents=doc)
        return chunks

    def persist(self,docs:List[Document]):
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=get_embedding(),
            persist_directory=self.db_path
            )
        return vectorstore
    
    def get_retriever(self):
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=get_embedding()
            ).as_retriever()
    

if __name__=='__main__':
    pass