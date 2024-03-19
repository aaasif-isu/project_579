import weaviate
import json
from embedding_util import generate_embeddings
from PyPDF2 import PdfReader
import os

client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
)

# Just to illustrate a simple Weaviate health check, if part of a larger system
print('is_ready:', client.is_ready())

file_path = '/Users/aaasif/OneDrive - Iowa State University/Spring 24/ComS 579/project/project_579/data/dummy.pdf'


# Class definition object. Weaviate's autoschema feature will infer properties
# when importing.
client.schema.delete_class('DocumentSearch')
class_obj = {
    "class": "DocumentSearch",
    "vectorizer": "none",
}

# Add the class to the schema
client.schema.create_class(class_obj)


# Test source documents
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

documents = read_and_clean_pdf(file_path)
chunks = chunk_text(documents)





# Configure a batch process. Since our "documents" is small, just setting the
# whole batch to the size of the "documents" list
client.batch.configure(batch_size=len(chunks))
with client.batch as batch:
    for i, doc in enumerate(chunks):
        print(f"document: {i}")

        properties = {
            "source_text": doc,
        }
        vector = generate_embeddings(doc)

        batch.add_data_object(properties, "DocumentSearch", vector=vector)

# test query
query = "apple"
query_vector = generate_embeddings(query)

# The default metric for ranking documents is by cosine distance.
# Cosine Similarity = 1 - Cosine Distance
result = client.query.get(
    "DocumentSearch", ["source_text"]
).with_near_vector(
    {
        "vector": query_vector,
        "certainty": 0.7
    }
).with_limit(2).with_additional(['certainty', 'distance']).do()

print(json.dumps(result, indent=4))

metainfo = client.get_meta()
#print(json.dumps(metainfo, indent=2))

#client.close()