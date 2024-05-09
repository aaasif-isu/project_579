import os
import torch
import weaviate
import warnings
import argparse
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import funix
import fitz
from typing import List
from funix.widget.builtin import BytesFile
from funix.hint import Markdown
import openai

load_dotenv()

client = None
Database_url=os.getenv("DB_URL")
openai_key= os.getenv("OPENAI_KEY")

def Upload_PDF(PDF_Files: List[BytesFile],
    ) -> Markdown:
   
    def read_pdf(PDF_Files):
        for file in PDF_Files:
            with fitz.open(stream=file, filetype="pdf") as doc:
                pdf_contents = [page.get_text() for page in doc] 

        return "\n\n".join(pdf_contents)

    def chunk_pdf(pdf_content):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(pdf_content)
        return chunks
            
    def insert(client, chunks, emb_model):
        response = Weaviate.from_texts(texts=chunks, embedding=emb_model,client=client)
        print("Inserted Successfully")
        return response
            

    def get_emb_mode():
        model = OpenAIEmbeddings(api_key=openai_key)
        return model
    
    try:
        pdf_contents = read_pdf(PDF_Files)    

        chunks = chunk_pdf(pdf_contents)

        emb_model = get_emb_mode()

        client = weaviate.Client(url=Database_url)

        global storage
        
        storage = insert(client, chunks, emb_model)


        message = "Vectors are stored in Database"

        return f"{message}"

    except Exception as e:
        return f"{e}"

def Query(Query : str) -> Markdown:
    model_name = "gpt-3.5-turbo-instruct"
    llm = OpenAI(api_key=openai_key,temperature=0.7, model_name=model_name)
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=storage.as_retriever(search_type="similarity",search_kwargs={'k':3})
    )

    answer = chain.invoke(Query)["result"]

    return f"{answer}"