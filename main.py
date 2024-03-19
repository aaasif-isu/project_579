import os
from langchain.embedders import HuggingFaceEmbedder
from weaviate import Client, ObjectsBatchRequest
from PyPDF2 import PdfReader

# Initialize the HuggingFace embedder
embedder = HuggingFaceEmbedder()

# Initialize Weaviate client
client = Client("http://localhost:8080")

def read_and_clean_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Remove page numbers
    text = '\n'.join([line for line in text.split('\n') if not line.isdigit()])
    return text

def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks):
    return [embedder.embed(chunk) for chunk in chunks]

def save_to_weaviate(embeddings):
    batch = ObjectsBatchRequest()
    for embedding in embeddings:
        # Create a data object for Weaviate
        data_object = {
            "content": embedding,
            "type": "Embedding"
        }
        batch.add(data_object)
    client.batch.create(batch)

if __name__ == "__main__":
    file_path = input("Enter the path to the PDF file: ")
    
    # Ensure the file exists
    if not os.path.isfile(file_path):
        print("File does not exist.")
        exit()

    text = read_and_clean_pdf(file_path)

    chunks = chunk_text(text)

    embeddings = embed_chunks(chunks)

    save_to_weaviate(embeddings)

    print("The PDF content has been indexed in Weaviate.")
