from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate


from db import Database
from utils import get_llm

class QA():
    def __init__(self, raw_path):
        self.raw_path = raw_path

    def qa_prompt(self):
        system_prompt = """
        Answer the given question only based on the given context. 
        If you do not know the answer, say you don't know.
        Do not make up answer on your own.
        Keep the answer short and crisp.

        Context:
        {context}
        """
        return ChatPromptTemplate(messages=[
            ('system',system_prompt),
            ('human',"{input}")
        ])

    def combine_docs_chain(self):
        chain = create_stuff_documents_chain(
            llm=get_llm(temperature=0.0),
            prompt = self.qa_prompt()
        )
        return chain

    def retrieval_chain(self):
        chain = create_retrieval_chain(
            retriever=Database(self.raw_path).get_retriever(),
            combine_docs_chain=self.combine_docs_chain() 
            ) 
        return chain
    
    def invoke(self, question):
        retrieval_chain = self.retrieval_chain()
        result = retrieval_chain.invoke({"input":question}) 
        return result['answer']
    