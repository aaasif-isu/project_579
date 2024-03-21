import warnings
import argparse
import weaviate
#from langchain_community.vectorstores import Weaviate 
#from weaviate.classes.config import Property, DataType
from langchain_community.document_loaders import PyPDFLoader   
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from sentence_transformers import SentenceTransformer

client=None
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def read_pdf(file_path):
    pdf_content = PyPDFLoader(file_path).load_and_split()
    return pdf_content

def chunk_pdf(pdf_content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = text_splitter.split_documents(pdf_content)
    return chunks

def gen_embeddings(chunks, source):
    vs = Weaviate.from_documents(documents=chunks, embedding=model, client=client)
    print("Indexing successfully")
    return vs


def start():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Read a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_content = read_pdf(args.pdf_file)
    chunks = chunk_pdf(pdf_content)
    vs = gen_embeddings(chunks, args.pdf_file)
    #ret = vs.as_retriever(search_type="similarity")
    #q = "What is attention?"
    #response = ret.invoke(q)
    #print(response)

try: 
    client = weaviate.Client(
        url="https://579-project-24szk15n.weaviate.network",
        auth_client_secret=weaviate.AuthApiKey("AoWIMOVxQzgfyHaVIAJMwXsg6mcTuxzeVoUr")
    )
    print("Connected to database successfully")
    start()
except Exception as e:
    print("Failed to connect to database!")
    print(e)
