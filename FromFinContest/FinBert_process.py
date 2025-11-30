# ran on google colab

import pandas as pd
import torch
from transformers import pipeline
import sys
import yfinance as yf

LOAD_FROM_FILE = False 
CSV_FILE_PATH = "TSLA_price_data.csv" # Expected columns: 'date', 'text'

# Output file name
OUTPUT_FILE = "data_sentiment.csv"

def setup_finbert():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    finbert = pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        device=device,
        top_k=None, # to compute neutral score better
        truncation=True,
    )
    return finbert

def get_scalar_sentiment(text, nlp_pipeline):
    """
    Converts text into a single number between -1.0 and +1.0.
    Formula: Probability(Positive) - Probability(Negative)
    """
    try:
        # Run inference
        # Output format [[{'label': 'Positive', 'score': 0.9}, ...]]
        results = nlp_pipeline(text)[0]
        
        # Convert list of dicts to a flat dictionary for easy lookup
        # e.g. {'Positive': 0.95, 'Negative': 0.01, 'Neutral': 0.04}
        scores = {item['label']: item['score'] for item in results}
        
        # Calculate the scalar signal
        # This tells the bot: "How positive is this relative to negative?"
        # Pure Neutral becomes 0.0 (because Pos and Neg will both be low)
        signal = scores.get('Positive', 0) - scores.get('Negative', 0)
        
        return signal
        
    except Exception as e:
        print(f"Error processing text: {str(e)[:50]}...")
        return 0.0 # Return Neutral

def download_data(ticker="TSLA", start="2018-01-01", end="2024-01-01"):

    output_filename = f"TSLA_price_data.csv"
    print(f"\nDownloading price data for {ticker}...")
    try:
        data = yf.download(ticker, start=start, end=end)
        data.to_csv(output_filename)
        print(f"Successfully saved")
    except Exception as e:
        print(f"An error occurred during download: {e}")


# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def main():

    download_data(ticker="TSLA", start="2018-01-01", end="2024-01-01")
    
    if LOAD_FROM_FILE:
        try:
            print(f"Reading data from {CSV_FILE_PATH}...")
            df = pd.read_csv(CSV_FILE_PATH)
            # Ensure column names match your data
            if 'text' not in df.columns:
                print("Error: CSV must have a 'text' column.")
                return
        except FileNotFoundError:
            print("File not found! Please check the path.")
            return
    else:
        print("Error, no data file specified.")
        return
    

    nlp = setup_finbert()

    print(f"Processing {len(df)} headlines...")
    
    df['sentiment_score'] = df['text'].apply(lambda x: get_scalar_sentiment(x, nlp))

    print("\nComplete! e.g.:")
    print(df[['date', 'sentiment_score', 'text']].head())
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved full results to: {OUTPUT_FILE}")
    

if __name__ == "__main__":
    main()