from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader 
from langchain_core.documents.base import Document

import os
import logging


class DocLoader():
    def __init__(self, doc_path):
        self.doc_path = doc_path 

    def load(self):
        ext = os.path.splitext(self.doc_path)[-1]
        if ext.lower()=='.pdf':
            return self.load_pdf()
        elif ext.lower() in {'.doc','.docx'}:
            return self.load_word() 
        else:
            logging.warning(f'Extension {ext} not supported; only pdf, doc, docx are supported') 

    def load_pdf(self)->Document:
        loader = PyPDFLoader(self.doc_path)
        return loader.load()

    def load_word(self)->Document:
        loader = UnstructuredWordDocumentLoader(self.doc_path)
        return loader.load()
