from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

model = joblib.load("sentiment_model.joblib")



app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class NewsInput(BaseModel):
    text: str



@app.post("/predict")
async def predict_sentiment(input: NewsInput):
    prediction = model.predict([input.text])[0]
    label = "negative" if prediction == 0 else "positive" 
    return {"label": label}