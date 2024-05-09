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
Database_url="https://579-project-upeahmau.weaviate.network"
openai_key= os.getenv("OPENAI_KEY")
#storage = None

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
        #model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        model = OpenAIEmbeddings(api_key=openai_key)
        return model
    
    # def initialize_model(model_name):
    # # token = os.getenv("HF_TOKEN")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name, 
    #         device_map="auto",  
    #         torch_dtype=torch.bfloat16,
    #     )
    #     #model = OpenAI(api_key=openai_key)
    #     return model
    
    # def initialize_tokenizer(model_name):
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    #     tokenizer.bos_token_id = 1 
    #     return tokenizer
    
    # def initialize_llm(model, tokenizer):
    #     llm = HuggingFacePipeline(pipeline=pipeline(
    #         "text-generation",
    #         model=model,
    #         tokenizer=tokenizer,
    #         device_map="auto",
    #         use_cache=True,
    #         max_length = 2048,
    #         do_sample=True,
    #         top_k=3,
    #         num_return_sequences=1,
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=tokenizer.eos_token_id
    #     ))

    #     return llm
    
    try:
        pdf_contents = read_pdf(PDF_Files)    

        chunks = chunk_pdf(pdf_contents)

        emb_model = get_emb_mode()

        client = weaviate.Client(url=Database_url)

        global storage
        
        storage = insert(client, chunks, emb_model)


        message = "Vectors are stored in Database"

        return f"{message}"

        # model_name = "Writer/palmyra-small"
        # # llm = OpenAI(temperature=0.7, model_name=model_name)
        # # memory = ConversationBufferMemory(
        # # memory_key='chat_history', return_messages=True)
        # # conversation_chain = ConversationalRetrievalChain.from_llm(
        # #     llm=llm,
        # #     chain_type="stuff",
        # #     retriever=storage.as_retriever(),
        # #     memory=memory
        # # )

        # model = initialize_model(model_name)

        # tokenizer = initialize_tokenizer(model_name)

        # llm = initialize_llm(model, tokenizer)

        # retrieve_answer = RetrievalQA.from_chain_type(
        #     llm=llm, chain_type="stuff", retriever=storage.as_retriever()
        # )

        # resp = retrieve_answer.run(Query)

        # # result = conversation_chain({"question": Query})
        # # answer = result["answer"]

        # return f"{resp}"

    except Exception as e:
        return f"{e}"

def Query(Query : str) -> Markdown:
    # def initialize_model(model_name):
    #     # token = os.getenv("HF_TOKEN")
    #     # model = AutoModelForCausalLM.from_pretrained(
    #     #     model_name, 
    #     #     device_map="auto",  
    #     #     torch_dtype=torch.bfloat16,
    #     # )
    #     model = OpenAI(api_key=openai_key)
    #     return model
    
    # def initialize_tokenizer(model_name):
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    #     tokenizer.bos_token_id = 1 
    #     return tokenizer
    
    # def initialize_llm(model, tokenizer):
    #     llm = HuggingFacePipeline(pipeline=pipeline(
    #         "text-generation",
    #         model=model,
    #         tokenizer=tokenizer,
    #         device_map="auto",
    #         use_cache=True,
    #         max_length = 2048,
    #         do_sample=True,
    #         top_k=3,
    #         num_return_sequences=1,
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=tokenizer.eos_token_id
    #     ))

    #     return llm

    model_name = "gpt-3.5-turbo-instruct"
    llm = OpenAI(api_key=openai_key,temperature=0.7, model_name=model_name)
    # memory = ConversationBufferMemory(
    # memory_key='chat_history', return_messages=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=storage.as_retriever(search_type="similarity",search_kwargs={'k':3})
    )

    answer = chain.invoke(Query)["result"]

    # model = initialize_model(model_name)

    # tokenizer = initialize_tokenizer(model_name)

    # llm = initialize_llm(model, tokenizer)

    # retrieve_answer = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=storage.as_retriever()
    # )

    # resp = retrieve_answer.run(Query)

    #result = conversation_chain({"question": Query})
    #answer = result["answer"]

    #print(result)

    return f"{answer}"