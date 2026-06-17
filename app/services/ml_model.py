import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

FEATURES = ['Return', 'MA5', 'price_v_ma5', 'Volatility', 'Vol_Change', 'Ret_5d']

FEATURE_LABELS = {
    'Return': 'recent daily return',
    'MA5': '5-day moving average',
    'price_v_ma5': 'price vs 5-day average',
    'Volatility': 'volatility',
    'Vol_Change': 'volume change',
    'Ret_5d': '5-day momentum',
}
def build_model():
    return make_pipeline(StandardScaler(),LogisticRegression())

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

    model = build_model()
    model.fit(X,y)

    return model

def explain_pred(model,latest,top_n=3):
    scaler = model.named_steps['standardscaler']
    clf = model.named_steps['logisticregression']

    x=latest[FEATURES].iloc[0].to_numpy(dtype=float)
    z=(x-scaler.mean_)/scaler.scale_
    contribs =clf.coef_[0] * z

    ranked = sorted(zip(FEATURES,contribs), key = lambda pair: abs(pair[1]),reverse=True)

    reasons = []

    for name,contrib in ranked[:top_n]:
        direction = "bullish" if contrib > 0 else "bearish"
        label= FEATURE_LABELS.get(name,name)
        reasons.append(f"{label}({direction})")
    return "; ".join(reasons)

def evaluate_model(df: pd.DataFrame,n_splits = 5):
    X = df[FEATURES]
    y = df['Target'].astype(int)

    scores = cross_val_score(build_model(), X, y, cv=TimeSeriesSplit(n_splits=n_splits), scoring='accuracy')

    model_accuracy = round(scores.mean()*100,2)
    baseline = round(max(y.mean(),1 - y.mean())*100,2)

    return model_accuracy,baseline

