import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf

def make_price_chart(ticker: str):
    df = yf.Ticker(ticker).history(period="6mo")
    if df.empty:
        return None

    close = df['Close']
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    fig,ax = plt.subplots(figsize= (10,5))
    ax.plot(df.index, close, label="Close", linewidth=1.5)
    ax.plot(df.index, ma5, label="MA5")
    ax.plot(df.index, ma20, label="MA20")
    ax.set_title(f"{ticker.upper()}-Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
