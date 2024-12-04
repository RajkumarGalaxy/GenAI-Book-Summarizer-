import os 
from pathlib import Path


from document_loader import DocLoader
from db import Database
from qa import QA 

class Process():
    def __init__(self):
        pass

    

if __name__=="__main__":
    raw_path = r"data\raw\the-book-thief-markus-zusak.pdf" 
    book_db = Database(raw_path=raw_path) 

    if not os.path.exists(book_db.db_path):
        document = DocLoader(doc_path=raw_path).load() 
        chunks = book_db.split(doc=document) 
        book_db.persist(docs=chunks) 
    
    question = input('Enter your question here:\n') 
    answer = QA(raw_path=raw_path).invoke(question=question)
    print(f'Answer: {answer}')