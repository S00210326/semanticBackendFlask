from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from elasticCloud import search  # Import your search function here
from elasticCloud import MSMarcoEncoder
import base64
import requests
from flask import Flask, request
from werkzeug.utils import secure_filename
from elasticsearch import Elasticsearch
from docx import Document

app = Flask(__name__)
CORS(app, origins=['http://localhost:4200'])

ELASTIC_PASSWORD = "WljB6OqBskWt9MRGV4GFBatA"

# Found in the 'Manage Deployment' page
CLOUD_ID = "SemanticElastic:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ3M2MzNzlmMDc0NTM0MTAzYTQzNDM4YmFkMTc1NzY2YSQ3YmVmYWVhZWY1MTc0YmJkOWM1MzczZmE5OWQ1MTIzMw=="

# Create the client instance
es_client = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)
#uses a MSMarcoEncoder which was built in the other python file. 
#USES MSMARCO-MINILM model at the moment for encoding
encoder = MSMarcoEncoder('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
model = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
index = "brandnew"

#Route for creating an index on the elastic-Cloud
@app.route('/createIndex', methods=['POST'])
def create_index():
    data = request.get_json()
    index_name = data.get('index_name')
    if not index_name:
        return jsonify({"error": "Missing 'index_name' parameter"}), 400

    config = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embeddings": {
                    "type": "dense_vector",
                    "dims": 384,  # Update this to the correct dimension of your embeddings
                    "index": False
                },
                
            }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        }
    }
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=config)
    return jsonify({"message": f"Index '{index_name}' created/already exists"}), 200



@app.route('/search', methods=['POST'])
def search_endpoint():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    # Modify your search function to return the results in a suitable format for jsonify()
    #Uses the search method from the other file
    result = search(query, es_client, model, index)  
    return jsonify(result)


#THIS IS FOR UPLOADING DOCX FILES
@app.route('/uploadDocument', methods=['POST'])
def upload_document():
    file = request.files['file']
    file.save(file.filename)

    # Assuming you're dealing with .docx files
    doc = Document(file.filename)
    
    # Split the document into chunks (paragraphs in this case)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip() != '']

    # Initialize an empty list to hold the document vectors
    doc_vectors = []

    # Encode each paragraph separately
    for paragraph in paragraphs:
        embeddings = encoder.encode(paragraph, max_length=512)
        doc_vectors.append(embeddings[0])

    # Index each paragraph separately
    for i, embeddings in enumerate(doc_vectors):
        es_client.index(index=index, body={
            "title": f"{file.filename} - paragraph {i+1}",
            "text": paragraphs[i],
            "embeddings": embeddings.tolist()
        })

    return jsonify({"message": "Document indexed"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
