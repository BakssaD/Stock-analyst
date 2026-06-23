from dotenv import load_dotenv

from app.services.chart import make_price_chart

load_dotenv()

from fastapi import FastAPI, HTTPException,Response
from pydantic import BaseModel

from app.services.ml_model import train_model, prep_data, explain_pred, evaluate_model
from app.services.news import get_headlines, summarize_news, generate_insight

class MarketData(BaseModel):
    price: float
    ma5: float
    volatility: float
    price_vs_ma5: float
    return_5d: float

class StockPrediction(BaseModel):
    ticker: str
    trend: str
    confidence: float
    model_accuracy: float
    baseline_accuracy: float
    market_data: MarketData
    reasoning: str
    news_summary:str
    final_insight: str

app=FastAPI()

@app.get("/")
def root():
    return{"message": "Stock analyst is running"}

@app.get("/stock/{ticker}/chart")
def stock_chart(ticker):
    buf=make_price_chart(ticker)
    if buf is None:
        raise HTTPException(status_code=404, detail=f"No data for '{ticker}'")
    return Response(content=buf.getvalue(),media_type="image/png")

@app.get("/stock/{ticker}")
def get_stock(ticker : str)->StockPrediction:
    train_df,latest, market_data=prep_data(ticker)

    if train_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    model=train_model(train_df)
    model_accuracy,baseline = evaluate_model(train_df)
    prob=model.predict_proba(latest)[0]

    class_probs = dict(zip(model.classes_,prob))
    prob_down = class_probs.get(0,0.0)
    prob_up = class_probs.get(1,0.0)

    prediction= 1 if prob_up > prob_down else 0
    trend= "Bullish" if prediction == 1 else "Bearish"
    confidence = round(max(prob_up,prob_down)*100,2)

    reasoning=explain_pred(model,latest)

    try:
        headlines = get_headlines(ticker)
        news_summary = summarize_news(ticker,headlines)
    except Exception:
        news_summary = "News summary unavailable."
    try:
        final_insight = generate_insight(ticker, trend, confidence, model_accuracy,baseline, reasoning, news_summary)
    except Exception:
        final_insight = "Insight unavailable."
    return StockPrediction(ticker = ticker.upper(),trend = trend,confidence = confidence,model_accuracy = model_accuracy, baseline_accuracy = baseline,market_data=market_data ,reasoning = reasoning,news_summary = news_summary , final_insight= final_insight)