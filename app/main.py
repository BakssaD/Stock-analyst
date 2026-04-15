from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from app.services.ml_model import train_model, prep_data, get_features

app=FastAPI()

@app.get("/")
def root():
    return{"message": "Stock analyst is running"}

@app.get("/stock/{ticker}")
def get_stock(ticker : str):
    try:
        df=prep_data(ticker)
        print(df.head())
        print(df.columns)
        model= train_model(df)
        features=get_features(df)

        prob=model.predict_proba([features])[0]

        prob_down = prob[0]
        prob_up = prob[1]

        prediction= 1 if prob_up > prob_down else 0
        trend= "Bullish" if prediction == 1 else "Bearish"
        confidance = round(max(prob_up,prob_down)*100,2)

        reasoning=[]

        if features['price_v_ma5'] > 1:
            reasoning.append("price above 5-day moving average")
        else:
            reasoning.append("price below 5-day moving average")

        if features['Ret_5d'] > 0:
            reasoning.append("positive 5-day momentum")
        else:
            reasoning.append("negative 5-day momentum")

        if features['Volatility'] > 0.02:
            reasoning.append("high volatility")
        else:
            reasoning.append("low volatility")

        reasoning = "; ".join(reasoning)

        data = {
            "ticker": ticker.upper(),
            "trend": trend,
            "confidance":confidance,
            "reasoning":reasoning
        }
        return jsonable_encoder(data)
    except Exception as e:
        return {"error": str(e)}