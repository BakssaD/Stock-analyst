import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression

def prep_data(ticker:str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    close = df['Close']
    df['Return'] = close.pct_change()
    df['MA5'] = close.rolling(5).mean()
    df['price_v_ma5'] = close / df['MA5']
    df['Volatility'] = close.rolling(5).std()
    df['Vol_Change'] = df['Volume'].pct_change()
    df['Ret_5d'] = close.pct_change(5)

    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    df = df.dropna()

    return df

def train_model(df: pd.DataFrame):

    X = df[['Return','MA5','price_v_ma5','Volatility','Vol_Change','Ret_5d']]
    y = df['Target']

    model = LogisticRegression()
    model.fit(X,y)

    return model

def get_features(df: pd.DataFrame):
    return df[['Return','MA5','price_v_ma5','Volatility','Vol_Change','Ret_5d']].iloc[-1]

