from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import os
import sys
import time
import logging
import traceback
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('chatbot2new.log')
handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# Log a message
logger.info('This is an info message')


app = FastAPI()

SEN_TR_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama3:8b"
emb = None
llm = None

system_prompt = """You are a helpful assistant. Given the following context and a question, generate an answer based on the context.
                    If the answer is not provided in the context kindly state "I dont know". Do not try to make things up.
                    Utilize multi-step reasoning to provide concise answers, focusing on key information.
                    CONTEXT : {context}
                    QUESTION : {input}"""

class DomianGPT(BaseModel):
    urls : List[str]
    question : str

llm = Ollama(model = "llama3")

def get_urls(urls : List[str]):
    logger.info("Loading URLs")
    data = ""
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if request was successful
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ')
            data += text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
    logger.info("data created successfully")
    return data

def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(data)
    logger.info("Text chunks created")
    return chunks

           
# Create the vector store and get Embeddings
def get_vectorstore(chunks):
    vectorstore = FAISS.from_texts(texts=chunks, embedding=emb) 
    logger.info("Vector store created")
    return vectorstore 

def get_conversational_chain():
    global doc_chain
    prompt = ChatPromptTemplate.from_template(system_prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    logger.info("Conversational chain created")
    return bool(doc_chain)

def rag_retrieval(vectorstore, ):
    retriever=vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,doc_chain)
    logger.info("RAG retrieval chain created")
    return retrieval_chain

def rag_pipeline(data):
    chunks = get_text_chunks(data)
    vector_store = get_vectorstore(chunks)
    retrieval_chain = rag_retrieval(vector_store)
    logger.info("RAG pipeline completed")
    return retrieval_chain

def load_transformer():
    global emb 
    emb= HuggingFaceEmbeddings(model_name=SEN_TR_MODEL)
    logger.info("Transformer model loaded")
    return bool(emb)

def load_model():
    global llm
    llm = Ollama(model=LLM_MODEL)
    logger.info("LLM model loaded")
    return bool(llm)

def init_system():
    logger.info("Initializing...")
    a = load_transformer()
    b = load_model()
    c = get_conversational_chain()
    logger.info(f"""load_transformer: {a} \n
               f"load_model: {b} \n
               f"get_conversational_chain: {c} \n""")
    if (a & b & c):
        logger.info("Initialization complete")
    else:
        logger.error("Initialization failed!")
    
start_time=time.time()
init_system()
end_time=time.time()
elapsed_time=end_time-start_time
logger.info(f"Time taken for init: {elapsed_time:.6f} seconds")

@app.get("/ping")
async def ping():
    return "hello"
        
@app.post("/process_urls")
async def process_urls(domain_gpt: DomianGPT):
    try:
        data = get_urls(domain_gpt.urls)
    except Exception as e:
        logger.error(f"Error processing URLs: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing URLs")
    
    # RAG pipeline building
    start_time=time.time()
    if data:
        retrieval_chain = rag_pipeline(data)
        logger.info(f"Retrieval Chain Type: {type(retrieval_chain)}")
    else:
        raise Exception("No data found in URL")
    logger.info("RAG pipeline created.")
    end_time=time.time()
    elapsed_time=end_time-start_time
    logger.info(f"Time taken for creating retrieval chain: {elapsed_time:.6f} seconds")
    
    # Ask the question
    result = retrieval_chain.invoke({"input": domain_gpt.question})
        
    return {
            "answer": result["answer"]
        }
    
    
    #if __name__ == "__main__":
    #    uvicorn.run(chatbot2, host = "localhost", port =8000)
