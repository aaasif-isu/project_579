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
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import funix
import fitz
from typing import List
from funix.widget.builtin import BytesFile
from funix.hint import Markdown

client = None

def RAG(Database_url: str,
        PDF_Files: List[BytesFile],
        chunk_size: int = 1000,
        Query: str="") -> Markdown:
   
    def read_pdf(PDF_Files):
        for file in PDF_Files:
            with fitz.open(stream=file, filetype="pdf") as doc:
                pdf_contents = [page.get_text() for page in doc] 

        return "\n\n".join(pdf_contents)

    # class chunk_pdf(object):
    #     def __init__(self, text: str, chunk_size):
    #         self.chunk_size = chunk_size
    #         self.current_token = 0
    #         self.text_tokenized = text.split()

    #     def __iter__(self):
    #         return self 

    #     def __next__(self):
    #         if self.current_token >= len(self.text_tokenized):
    #             raise StopIteration
    #         else:
    #             chunk = self.text_tokenized[self.current_token : self.current_token + self.chunk_size]
    #             self.current_token += self.chunk_size
    #             return " ".join(chunk)

    def chunk_pdf(pdf_content, chunk_size):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(pdf_content)
        return chunks
            
    def insert(client, chunks, emb_model):
        response = Weaviate.from_texts(texts=chunks, embedding=emb_model,client=client)
        print("Inserted Successfully")
        return response
            

    def get_emb_mode():
        emb_model = os.getenv("EMB_MODEL")
        model = HuggingFaceEmbeddings(model_name=emb_model)
        return model
    
    def initialize_model(model_name):
        token = os.getenv("HF_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",  
            torch_dtype=torch.bfloat16,
            token=token,
        )
        return model
    
    def initialize_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
        tokenizer.bos_token_id = 1 
        return tokenizer
    
    def initialize_llm(model, tokenizer):
        llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            use_cache=True,
            max_length = 2048,
            do_sample=True,
            top_k=3,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        ))

        return llm
    
    try:
        pdf_contents = read_pdf(PDF_Files)    

        chunks = chunk_pdf(pdf_contents, chunk_size)

        emb_model = get_emb_mode()

        client = weaviate.Client(url=Database_url)

        storage = insert(client, chunks, emb_model)

        model_name = os.getenv("MODEL")

        model = initialize_model(model_name)

        tokenizer = initialize_tokenizer(model_name)

        llm = initialize_llm(model, tokenizer)

        retrieve_answer = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=storage.as_retriever()
        )

        resp = retrieve_answer.run(Query)

        return f"{resp}"

    except Exception as e:
        return f"{e}"
    

    



