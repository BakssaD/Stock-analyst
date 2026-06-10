from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from app.services.ml_model import train_model, prep_data

class StockPrediction(BaseModel):
    ticker: str
    trend: str
    confidence: float
    reasoning: str

app=FastAPI()

@app.get("/")
def root():
    return{"message": "Stock analyst is running"}

@app.get("/stock/{ticker}")
def get_stock(ticker : str)->StockPrediction:
    train_df,latest=prep_data(ticker)

    if train_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    model=train_model(train_df)
    prob=model.predict_proba(latest)[0]

    class_probs=dict(zip(model.classes_,prob))
    prob_down = class_probs.get(0,0.0)
    prob_up = class_probs.get(1,0.0)

    prediction= 1 if prob_up > prob_down else 0
    trend= "Bullish" if prediction == 1 else "Bearish"
    confidance = round(max(prob_up,prob_down)*100,2)

    row=latest.iloc[0]
    reasoning=[]

    if row['price_v_ma5'] > 1:
        reasoning.append("price above 5-day moving average")
    else:
        reasoning.append("price below 5-day moving average")
    if row['Ret_5d'] > 0:
        reasoning.append("positive 5-day momentum")
    else:
        reasoning.append("negative 5-day momentum")
    if row['Volatility'] > 0.02:
        reasoning.append("high volatility")
    else:
        reasoning.append("low volatility")
        reasoning = "; ".join(reasoning)

    return StockPrediction(ticker = ticker.upper(),trend = trend,confidence = confidance,reasoning = reasoning)