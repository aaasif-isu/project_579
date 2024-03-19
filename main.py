import argparse
from langchain_community.vectorstores import Weaviate 
from langchain_community.document_loaders import PyPDFLoader   
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

import weaviate

client=None
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

schema = {
    "classes": [
        {
            "class": "Embeddings",
            "properties": [
                {
                    "property": "source",
                    "dataType": "string"
                },
                {
                    "property": "index",
                    "dataType": "int"
                },
                {
                    "property": "embedding",
                    "dataType": "vector"
                }
            ]
        }
    ]
}

def save_to_db(source, index, embedding):
    client.batch_import(
        "Embeddings",
        [
            {
                "source": source,
                "index": index,
                "embedding": embedding
            }
        ]
    )
    print("Inserted")

def read_and_filter_pdf(file_path):
    pdf_content = PyPDFLoader(file_path).load_and_split()[0].page_content
    text = '\n'.join([line for line in pdf_content.split('\n') if not line.isdigit()])
    return text

def chunk_pdf(pdf_content):
    text_splitter = CharacterTextSplitter(
        separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len
    )
    chunks = text_splitter.split_text(pdf_content)
    return chunks

def gen_embeddings(chunks, source):
    for i, chunk in chunks:
        embedding = model.encode(chunk)
        save_to_db(source, i, embedding)
        #print(embedding)

def start():
    parser = argparse.ArgumentParser(description="Read a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_content = read_and_filter_pdf(args.pdf_file)
    #print("PRINTING RAW")
    #print(pdf_content)
    #print("PRINTING CHUNKS")
    chunks = chunk_pdf(pdf_content)
    #print(chunks)
    gen_embeddings(chunks, args.pdf_file)

try: 
    client = weaviate.Client(
        url="https://579-project-sxs0vdgu.weaviate.network",
    )
    client.schema.create_class(schema)
    start()
except Exception as e:
    print("Failed to connect to database!")
    print(e)
finally:
    client.close()




