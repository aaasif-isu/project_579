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

# def save_to_db(source, index, embedding):
#     embeddings.data.insert(
#         properties=
#             {
#                 "source": source,
#                 "index": index,
#                 "embedding": embedding
#             }
#     )
#     print("Inserted")

def read_pdf(file_path):
    pdf_content = PyPDFLoader(file_path).load_and_split()
    return pdf_content

def chunk_pdf(pdf_content):
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len
    # )
    # chunks = text_splitter.split_text(pdf_content)
    # return chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = text_splitter.split_documents(pdf_content)
    return chunks

def gen_embeddings(chunks, source):
    vs = Weaviate.from_documents(documents=chunks, embedding=model, client=client)
    return vs


def start():
    parser = argparse.ArgumentParser(description="Read a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_content = read_pdf(args.pdf_file)
    # print("PRINTING RAW")
    # print(pdf_content)
    # print("PRINTING CHUNKS")
    chunks = chunk_pdf(pdf_content)
    # print(chunks[4])
    vs = gen_embeddings(chunks, args.pdf_file)
    ret = vs.as_retriever(search_type="similarity")
    q = "What is attention?"
    response = ret.invoke(q)
    print(response)

try: 
    client = weaviate.Client(
        url="https://579-project-24szk15n.weaviate.network",
        auth_client_secret=weaviate.AuthApiKey("AoWIMOVxQzgfyHaVIAJMwXsg6mcTuxzeVoUr")
    )
    # client.collections.create("Embeddings",
    #     properties=[
    #         Property(name="source", data_type=DataType.TEXT),
    #         Property(name="index", data_type=DataType.INT),
    #         Property(name="embedding", data_type=DataType.NUMBER_ARRAY),
    #     ]
    # )
    #client.collections.delete("Embeddings")
    #embeddings = client.collections.get("Embeddings")
    start()
except Exception as e:
    print("Failed to connect to database!")
    print(e)




