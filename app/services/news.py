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