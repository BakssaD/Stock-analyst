import os

import yfinance as yf
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_headlines(ticker: str, limit: int = 5)->list[str]:
    stock=yf.Ticker(ticker)
    news = stock.news or []
    headlines = []
    for item in news[:limit]:
        title = item.get("content", {}).get("title") or item.get("title")
        if title:
            headlines.append(title)
    return headlines

def summarize_news(ticker: str, headlines: list[str])->str:
    if not headlines:
        return ("No recent news available.")

    joined = "\n".join((f"- {h}" for h in headlines))
    prompt = (f"Here are recent news headlines for {ticker}:\n{joined}\n\n"
        "In 2-3 sentences, summarize the overall news sentiment "
        "(positive, negative, or mixed) and the main themes. "
        "Do not give financial advice.")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content.strip()

def generate_insight(ticker, trend, confidence, model_accuracy,baseline_accuracy, reasoning, news_summary):
    edge = model_accuracy - baseline_accuracy
    prompt = ( f"You are a cautious stock analyst assistant. Write ONE concise paragraph "
        f"(3-4 sentences) giving an overall outlook for {ticker}. Synthesize the "
        f"technical signal and the news. Be balanced, acknowledge uncertainty, and "
        f"do NOT give buy/sell financial advice.\n\n"
        f"ML signal: {trend} (model confidence {confidence}%)\n"
        f"Model accuracy: {model_accuracy}% vs naive baseline {baseline_accuracy}% "
        f"(edge of only {edge:.2f} points, so the model has little real predictive power)\n"
        f"Key technical factors: {reasoning}\n"
        f"Recent news: {news_summary}\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()
