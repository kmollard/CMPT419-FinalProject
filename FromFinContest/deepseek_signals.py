import os
import json
import pandas as pd
from openai import OpenAI
import time


client = OpenAI(
    # It's recommended to use an environment variable for your API key
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

if not client.api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set. Please set it to your API key.")

COUNT = 0
##### Prompt Engineering
# A robust prompt that requests a JSON object with all the required fields.
# It includes a high-quality example to make the output very reliable.
SENTIMENT_PROMPT_TEMPLATE = """
You are a financial news analyst. Your task is to analyze the sentiment of the following news article.

You must provide your analysis in a structured JSON format. The JSON object must contain the following keys:
- "sentiment_score": An integer from 1 (extremely negative) to 5 (extremely positive), where 3 means neutral.
- "confidence_score_sentiment": A float between 0.0 and 1.0, representing your confidence in the sentiment analysis.
- "reasoning_sentiment": A concise, one-sentence explanation for your sentiment score.

---
Here is a perfect example of the output format:
{{
    "sentiment_score": 4,
    "confidence_score_sentiment": 0.95,
    "reasoning_sentiment": "The article reports a significant earnings beat and a positive future outlook, which are strong bullish signals."
}}
---

Now, analyze the following news item and provide ONLY the JSON object as your response.

Title: {title}
Article Text: {text}
"""

RISK_PROMPT_TEMPLATE = """
You are a professional cryptocurrency risk analyst. Your task is to analyze the following news article to identify potential risks related to Bitcoin (BTC) or the broader crypto market.
\n
You must provide your analysis in a structured JSON format. The JSON object must contain the following keys:
- "risk_score": An integer from 1 (low risk) to 5 (high risk).
- "confidence_score_risk": A float between 0.0 and 1.0, representing your confidence in the risk analysis.
- "reasoning_risk": A concise, one-sentence explanation for your risk assessment.

---
Here is a perfect example of the output format for a BTC-related article:
{{
    "risk_score": 4,
    "confidence_score_risk": 0.85,
    "reasoning_risk": "The announcement of new government regulations on crypto transactions introduces significant legal and compliance uncertainty, potentially stifling adoption."
}}
---

Now, analyze the following news item and provide ONLY the JSON object as your response.

Title: {title}
Article Text: {text}
"""

def analyze_article_sentiment(title: str, text: str) -> dict | None:
    """
    Calls the DeepSeek API to get a structured sentiment analysis.

    Args:
        title: The title of the news article.
        text: The content of the news article.

    Returns:
        A dictionary containing 'sentiment_score', 'confidence_score_sentiment', and 'reasoning_sentiment',
        or None if an error occurs.
    """
    formatted_prompt = SENTIMENT_PROMPT_TEMPLATE.format(title=title, text=text)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0,  # Deterministic output
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        analysis_data = json.loads(response_content)

        required_keys = ["sentiment_score", "confidence_score_sentiment", "reasoning_sentiment"]
        if all(key in analysis_data for key in required_keys):
            return analysis_data
        else:
            print(f"Error: Missing one or more required keys in response for title: '{title}'")
            return None

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON for title: '{title}'. Raw response: {response_content}")
        return None
    except Exception as e:
        print(f"An API error occurred for title '{title}': {e}")
        return None

def analyze_article_risk(title: str, text: str) -> dict | None:
    """
    Calls the DeepSeek API to get a structured risk analysis.

    Args:
        title: The title of the news article.
        text: The content of the news article.

    Returns:
        A dictionary containing 'risk_score', 'confidence_score_risk', and 'reasoning_risk',
        or None if an error occurs.
    """
    formatted_prompt = RISK_PROMPT_TEMPLATE.format(title=title, text=text)
    global COUNT
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0,  # Deterministic output
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        analysis_data = json.loads(response_content)

        required_keys = ["risk_score", "confidence_score_risk", "reasoning_risk"]
        COUNT += 1
        print(f"Processed {COUNT} articles...")
        if all(key in analysis_data for key in required_keys):
            return analysis_data
        else:
            print(f"Error: Missing one or more required keys in response for title: '{title}'")
            return None

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON for title: '{title}'. Raw response: {response_content}")
        return None
    except Exception as e:
        print(f"An API error occurred for title '{title}': {e}")
        return None
    
def get_sentiment(row: pd.Series) -> pd.Series:

    analysis = analyze_article_sentiment(row['title'], row['article_text'])
    
    if analysis:
        return pd.Series(analysis)
    else:
        return pd.Series({
            'sentiment_score': None,
            'confidence_score_sentiment': None,
            'reasoning_sentiment': None
        })

def get_risk(row: pd.Series) -> pd.Series:

    analysis = analyze_article_risk(row['title'], row['article_text'])
    
    if analysis:
        return pd.Series(analysis)
    else:
        return pd.Series({
            'risk_score': None,
            'confidence_score_risk': None,
            'reasoning_risk': None
        })


# --- 3. Data Loading and Processing ---
if __name__ == "__main__":
    news_df = pd.read_csv('./datasets/combined_data.csv')  # Replace with your actual CSV file path
    # print("Start sentiment analysis...")
    # sentiment_results_df = news_df.apply(get_sentiment, axis=1)
    # enriched_df = news_df.join(sentiment_results_df)
    # enriched_df.to_csv('news_with_sentiment_analysis.csv', index=False, encoding='utf-8')

    print("Start risk analysis...")
    risk_results_df = news_df.apply(get_risk, axis=1)
    enriched_df = news_df.join(risk_results_df)
    enriched_df.to_csv('news_with_risk_analysis.csv', index=False, encoding='utf-8')

    # # Save the final DataFrame to a new CSV file
    # output_filename = "news_with_sentiment_analysis.csv"
    # enriched_df.to_csv(output_filename, index=False, encoding='utf-8')
    # print(f"\nSuccessfully saved the enriched data to '{output_filename}'")
