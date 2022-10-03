#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from importlib.resources import contents
from elasticsearch import Elasticsearch
import json, requests

# create a client instance of Elasticsearch
elastic_client = Elasticsearch(hosts="http://localhost:9200")

# create a Python dictionary for the search query:
search_param = {
    "query": {
        "terms": {
            "_id": [ 1 ] # find Ids '1234' and '42'
        },
        "default_field": "content"
    }
}

# search_param_2  = {
#    "query": {
#         "terms": {
#             "_id": [ 1 ] # find Ids '1234' and '42'
#         },
#         "default_field": "content"
#     }

# }

# get a response from the cluster
response = elastic_client.search(index="test-index", body=search_param)
print ('response:', response)



# some_string = '{"field1" : "fine me!"}'

# # turn a JSON string into a dictionary:
# some_dict = json.loads(some_string)

# # Python dictionary object representing an Elasticsearch JSON query:
# search_param = {
#     'query': {
#         'match': {
#             'field1': 'find me!'
#         }
#     }
# }

# # get another response from the cluster
# response = elastic_client.search(index="some_index", body=search_param)
# print ('response:', response)