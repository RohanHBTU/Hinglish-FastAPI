from fastapi import FastAPI
from inference import outcome
import warnings
warnings.filterwarnings("ignore")

app=FastAPI()

@app.get('/')
def index():
    return {"message":"Welcome to Hinglish Translator"}

@app.post('/predict')
def predict(input: str):
    return {"result":outcome(input)}
