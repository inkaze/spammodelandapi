from datetime import datetime
import re
import json
from elasticsearch import Elasticsearch
es = Elasticsearch(hosts="http://localhost:9200")

# doc = {
#     'author': 'kimchy',
#     'text': 'Elasticsearch: cool. bonsai cool.',
#     'timestamp': datetime.now(),
# }
# resp = es.index(index="test-index", id=1, document=doc)

def es_getString():
 resp = es.get(index="test-index", id=2)

 es.indices.refresh(index="test-index")

 data = str(resp['_source']["text"])
 
 return data