import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['Return', 'MA5', 'price_v_ma5', 'Volatility', 'Vol_Change', 'Ret_5d']

def prep_data(ticker:str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    close = df['Close']
    df['Return'] = close.pct_change()
    df['MA5'] = close.rolling(5).mean()
    df['price_v_ma5'] = close / df['MA5']
    df['Volatility'] = df['Return'].rolling(5).std()
    df['Vol_Change'] = df['Volume'].pct_change()
    df['Ret_5d'] = close.pct_change(5)

    next_ret=df['Return'].shift(-1)
    df['Target']=(next_ret>0).astype('Int64')
    df.loc[next_ret.isna(),'Target']=pd.NA

    df = df.dropna(subset=FEATURES)

    train_df=df[df['Target'].notna()]
    latest=df[FEATURES].iloc[[-1]]


    return train_df,latest

def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['Target'].astype(int)

    model = make_pipeline(StandardScaler(),LogisticRegression())
    model.fit(X,y)

    return model

