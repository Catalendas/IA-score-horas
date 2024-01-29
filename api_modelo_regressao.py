from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

app = FastAPI()

class request_body(BaseModel): 
        horas_estudo: float

modelo_pontuacao = joblib.load("./modelo_regressao.pkl")

@app.post('/predict')
def predict(data: request_body):
    input_feature = [[data.horas_estudo]]

    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {"pontuação_teste": y_pred.tolist()}