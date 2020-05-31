from pathlib import Path
import pandas as pd
from datetime import date

predictions_file = Path('GetStockPredictions/predictions.pkl')
predictions_df = pd.read_pickle(predictions_file)


def get_values(cols, ticker=None, rdate=None):
    if not ticker and not rdate:
      raise ValueError('at least one of ticker or rdate should be provided')
    df_filter = [True]*len(predictions_df)
    if ticker:
      df_filter = predictions_df['ticker'] == ticker
    if rdate:
      df_filter = (predictions_df['created_at'] == rdate) & df_filter
    df_subset = predictions_df.loc[df_filter, cols]
    return df_subset


if __name__ == '__main__':
    rdate = date(2020, 1, 13)
    ticker='AAPL'
    print(f'TESTING:\nSentiment for {ticker} on {rdate}:')
    aapl_sentiment = get_values(cols=['bearish_score_mean', 'bullish_score_mean'],ticker=ticker, rdate=rdate)
    print(aapl_sentiment)


