from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if __name__ == "__main__":
    text = input("Enter a sentence for sentiment analysis: ")

    textblob_result = analyze_sentiment_textblob(text)
    vader_result = analyze_sentiment_vader(text)

    print(f"TextBlob Sentiment: {textblob_result}")
    print(f"VADER Sentiment: {vader_result}")
