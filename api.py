import string
from fastapi import FastAPI
# import pathlib
# import namnn
import spam

app = FastAPI()

@app.get("/test/{text}")
def test(text):
  return {spam.Get_Result(text)}

  


