from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import retrieval_qa

import os 
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


from document_loader import DocLoader

class Process():
    def __init__(self):
        load_dotenv() 
        self.api_key = os.getenv('GOOGLE_API_KEY')
    def get_llm(self, temperature):
        return ChatGoogleGenerativeAI(
            model='gemini-1.5-pro', 
            goole_api_key=self.api_key,
            temperature=temperature,
            max_retries=2,
            )
    
    





if __name__=='__main__':
    # test llm
    llm = Process().get_llm(temperature=0.5)
    res = llm.invoke("How to create Docker image for python project?").content
    print(res)

    # load data
    data_dir = Path('data/raw') 
    sample_file = sorted(data_dir.iterdir())[-1]
    print(sample_file)

    doc = DocLoader(sample_file).load()
    if doc != None:
        print(type(doc[0]))
        print(doc[10])
