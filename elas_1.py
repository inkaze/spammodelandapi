from datetime import datetime
from email import message
from http.client import REQUEST_TIMEOUT
from importlib.resources import path
import pathlib
import string
from tokenize import String
from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200')


def put_elasticsearch(index_name :string , id_mess : string , content : string ) :
    resp = es.index(index=index_name, id=id_mess, document=content,REQUEST_TIMEOUT = 40)
    # print(resp['result'])
 

def get_elasticsearch(index_name :string , id_mess : string , content : string) :
    resp = es.get(index=index_name, id=id_mess)
    return resp['_source']
    # print(resp['_source'])

def excute():
    put_elasticsearch("test_index","2","em ơi đi thôi trời tối rồi đấy")
if __name__ == "__main__":
    excute()
