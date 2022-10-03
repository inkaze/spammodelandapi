from fastapi import FastAPI
import pathlib
import namnn
import spam

app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

@app.get("/")
def index():
  pred= spam.Get_Result()
  return(pred)  

# @app.on_event("startup")
# def start_model():
  


