
import os
import json

from sympy import true, false
from tqdm import tqdm
from torchvision.models.vision_transformer import Encoder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class MSMarcoEncoder:
    def __init__(self, model_name: str, device : str='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

    def encode(self, text:str, max_length: int):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs, return_dict=True)
        # Perform pooling
        embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class JsonlCollectionIterator:
    """
    adapted from https://github.com/castorini/pyserini/blob/master/pyserini/encode/_base.py
    """

    def __init__(self, collection_path: str, fields=None, delimiter="\n"):
        if fields:
            self.fields = fields
        else:
            self.fields = ['text']
        self.delimiter = delimiter
        self.all_info = self._load(collection_path)
        self.size = len(self.all_info['id'])
        self.batch_size = 1
        self.shard_id = 0
        self.shard_num = 1

    def __call__(self, batch_size=1, shard_id=0, shard_num=1):
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.shard_num = shard_num
        return self

    def __iter__(self):
        total_len = self.size
        shard_size = int(total_len / self.shard_num)
        start_idx = self.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_len)
        if self.shard_id == self.shard_num - 1:
            end_idx = total_len
        to_yield = {}
        for idx in tqdm(range(start_idx, end_idx, self.batch_size)):
            for key in self.all_info:
                to_yield[key] = self.all_info[key][idx: min(idx + self.batch_size, end_idx)]
            yield to_yield

    def _parse_fields_from_info(self, info):
        """
        :params info: dict, containing all fields as speicifed in self.fields either under
        the key of the field name or under the key of 'contents'.  If under `contents`, this
        function will parse the input contents into each fields based the self.delimiter
        return: List, each corresponds to the value of self.fields
        """
        n_fields = len(self.fields)

        # if all fields are under the key of info, read these rather than 'contents'
        if all([field in info for field in self.fields]):
            return [info[field].strip() for field in self.fields]

        assert "contents" in info, f"contents not found in info: {info}"
        contents = info['contents']
        # whether to remove the final self.delimiter (especially \n)
        # in CACM, a \n is always there at the end of contents, which we want to remove;
        # but in SciFact, Fiqa, and more, there are documents that only have title but not text (e.g. "This is title\n")
        # where the trailing \n indicates empty fields
        if contents.count(self.delimiter) == n_fields:
            # the user appends one more delimiter to the end, we remove it
            if contents.endswith(self.delimiter):
                # not using .rstrip() as there might be more than one delimiters at the end
                contents = contents[:-len(self.delimiter)]
        return [field.strip(" ") for field in contents.split(self.delimiter)]

    def _load(self, collection_path):
        filenames = []
        if os.path.isfile(collection_path):
            filenames.append(collection_path)
        else:
            for filename in os.listdir(collection_path):
                filenames.append(os.path.join(collection_path, filename))
        all_info = {field: [] for field in self.fields}
        all_info['id'] = []
        for filename in filenames:
            with open(filename) as f:
                for line_i, line in tqdm(enumerate(f)):
                    info = json.loads(line)
                    _id = info.get('_id', info.get('docid', None))
                    if _id is None:
                        raise ValueError(f"Cannot find 'id' or 'docid' from {filename}.")
                    all_info['id'].append(str(_id))
                    fields_info = self._parse_fields_from_info(info)
                    if len(fields_info) != len(self.fields):
                        raise ValueError(
                            f"{len(fields_info)} fields are found at Line#{line_i} in file {filename}." \
                            f"{len(self.fields)} fields expected." \
                            f"Line content: {info['contents']}"
                        )

                    for i in range(len(fields_info)):
                        all_info[self.fields[i]].append(fields_info[i])
        return all_info



from elasticsearch import Elasticsearch

ELASTIC_PASSWORD = "WljB6OqBskWt9MRGV4GFBatA"

# Found in the 'Manage Deployment' page
CLOUD_ID = "SemanticElastic:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ3M2MzNzlmMDc0NTM0MTAzYTQzNDM4YmFkMTc1NzY2YSQ3YmVmYWVhZWY1MTc0YmJkOWM1MzczZmE5OWQ1MTIzMw=="

# Create the client instance
# es_client = Elasticsearch(
#     cloud_id=CLOUD_ID,
#     basic_auth=("elastic", ELASTIC_PASSWORD)
# )
# config = {
#     "mappings": {
#         "properties": {
#             "title": {"type": "text"},
#             "text": {"type": "text"},
#             "embeddings": {
#                 "type": "dense_vector",
#                 "dims": 384,  # Update this to the correct dimension of your embeddings
#                 "index": False
# },

#             }
#     },
#     "settings": {
#         "number_of_shards": 2,
#         "number_of_replicas": 1
#     }
# }
# index_name = "testpython"
# if not es_client.indices.exists(index=index_name):
#     # Create the index if it does not exist
#     es_client.indices.create(
#         index=index_name,
#         settings=config["settings"],
#         mappings=config["mappings"],
#     )




# #check if the index has been created successfully
# print(es_client.indices.exists(index=[index_name]))

#
collection_path = 'C:/Users/pgonigle/PycharmProjects/AILearning/msmarco/corpus2.jsonl'
collection_iterator = JsonlCollectionIterator(collection_path, fields=['title', 'text'])
encoder = MSMarcoEncoder('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

def index_documents(collection_iterator, encoder, es_client, index_name):
    for batch_info in collection_iterator(batch_size=256, shard_id=0, shard_num=1):
        embeddings = encoder.encode(batch_info['text'], 512)
        batch_info["dense_vectors"] = embeddings

        actions = []
        for i in range(len(batch_info['id'])):
            action = {"index": {"_index": index_name, "_id": batch_info['id'][i]}}
            doc = {
                "title": batch_info['title'][i],
                "text": batch_info['text'][i],
                "embeddings": batch_info['dense_vectors'][i].tolist()
            }
            actions.append(action)
            actions.append(doc)

        es_client.bulk(index=index_name, operations=actions)

    result = es_client.count(index=index_name)

    # print the total number of documents in the index
    print(result.body['count'])

# INDEX ALL OF THE DOCUMENTS FROM THE JSON FIELD THAT WERE EMBEDDED
#index_documents(collection_iterator, encoder, es_client,index_name)


# print(es_client.get(index=["msmarco-demo"], id="0", request_timeout=60))
def search(query: str, es_client: Elasticsearch, model: str, index: str, top_k: int = 10):
    encoder = MSMarcoEncoder(model)
    #query gets encoded and is checked with cosSim against the document
    query_vector = encoder.encode(query, max_length=64)
    query_dict = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'embeddings') + 1.0",
                    "params": {
                        "queryVector": query_vector[0].tolist()
                    }
                }
            }
        },
        "size": top_k,
        #returns
        "_source": ["title", "text", "id"]
    }
    res = es_client.search(index=index, body=query_dict)

    #PRINTING HITS INSIDE THE CONSOLE
    for hit in res["hits"]["hits"]:
        print(hit)
        print(f"Document ID: {hit['_id']}")
        print(f"Document Title: {hit['_source']['title']}")
        print(f"Document Text: {hit['_source']['text']}")
        print("=======================================================\n")
    #ACTUALLY RETURNING HITS IN CLIENT
    hits = []
    for hit in res["hits"]["hits"]:
        hits.append({
            "id": hit["_id"],
            "title": hit["_source"]["title"],
            "text": hit["_source"]["text"]
        })
    return hits


  



if __name__ == "__main__":

  search(query="Music",
         es_client=es_client,
         model="sentence-transformers/msmarco-MiniLM-L6-cos-v5",
         index=index_name)