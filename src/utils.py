from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os 
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

def get_llm(temperature):
    return ChatGoogleGenerativeAI(
        model='gemini-1.5-pro', 
        goole_api_key=api_key,
        temperature=temperature,
        max_retries=2,
        )

def get_embedding():
    return GoogleGenerativeAIEmbeddings(model='models/embedding-001',google_api_key=api_key) 


if __name__=='__main__':
    # test llm
    llm = get_llm(temperature=0.5)
    res = llm.invoke("How to create Docker image for python project?").content
    print(res)