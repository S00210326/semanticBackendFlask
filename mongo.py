from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticCloud import MSMarcoEncoder
# Replace with your MongoDB connection string
MONGO_URI = 'mongodb+srv://pgonigle1:bnwZCOWxSRtbgvMC@proteins.x4ubsnu.mongodb.net/proteins'
MONGO_DB_NAME = 'proteins'
MONGO_COLLECTION_NAME = 'proteins'

# Replace with your Elasticsearch connection string
ELASTIC_INDEX = 'brandnew'


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    return client, collection

def get_elastic_client():
    ELASTIC_PASSWORD = "WljB6OqBskWt9MRGV4GFBatA"
    CLOUD_ID = "SemanticElastic:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ3M2MzNzlmMDc0NTM0MTAzYTQzNDM4YmFkMTc1NzY2YSQ3YmVmYWVhZWY1MTc0YmJkOWM1MzczZmE5OWQ1MTIzMw=="

    client = Elasticsearch(
        cloud_id=CLOUD_ID,
        http_auth=("elastic", ELASTIC_PASSWORD)
    )
    return client

def main():
    mongo_client, collection = get_mongo_collection()
    elastic_client = get_elastic_client()
    encoder = MSMarcoEncoder('sentence-transformers/msmarco-MiniLM-L6-cos-v5') 

    # Read data from MongoDB
    for doc in collection.find({}):
        id_ = str(doc.pop('_id'))  # Convert ObjectId to string and remove from doc

        # Concatenate fields into a single string #QUITE STRICT HOWEVER AS IT IS THE EXACT FIELDS FOR THE DATA
        text = f"Protein Source: {doc['protein_source']}, Subclass: {doc['subclass']}, Food: {doc['food']}"


        # Generate embeddings for the text
        embeddings = encoder.encode(text, max_length=512)[0]

        # Prepare Elasticsearch document
        es_doc = {
            "title": id_,  # Use the id as title, or any other title
            "text": text,
            "embeddings": embeddings.tolist()
        }

        # Insert data into Elasticsearch
        elastic_client.index(index=ELASTIC_INDEX, id=id_, body=es_doc)
        print(f"Indexed document with ID: {id_}")

    # Close MongoDB connection
    mongo_client.close()

if __name__ == "__main__":
    main()
