import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt


class WebContentAggregator:
    def __init__(self):
        self.search_results = []

    def dynamic_search_query(self, query):
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0;Win64)"}
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.find_all("a")
            self.search_results = [result.get("href")
                                   for result in search_results]
        except requests.exceptions.RequestException as e:
            print(f"Error in dynamic search query: {e}")

    def web_scraping(self):
        scraped_data = []
        for url in self.search_results:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                content = soup.get_text()
                scraped_data.append(content)
            except requests.exceptions.RequestException as e:
                print(f"Error scraping URL: {url}, {e}")
        return scraped_data


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.nlp_pipeline = pipeline(
            "text-classification", model="textattack/roberta-base-imdb")

    def analyze_sentiment(self, text):
        sentiment_scores = self.sia.polarity_scores(text)
        sentiment = max(sentiment_scores, key=sentiment_scores.get)
        return sentiment

    def extract_keywords(self, text):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text.lower())
        freq_dist = nltk.FreqDist(tokens)
        return freq_dist.most_common(5)


class ContentCategorization:
    def __init__(self):
        self.model = MultinomialNB()
        self.pipeline = Pipeline(
            [("vectorizer", CountVectorizer()), ("model", self.model)])

    def train_model(self, texts, categories):
        self.pipeline.fit(texts, categories)

    def predict_category(self, text):
        predicted_category = self.pipeline.predict([text])
        return predicted_category[0]


class StockPrediction:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = MinMaxScaler()

    def load_data(self, file_name):
        data = pd.read_csv(file_name)
        return data

    def preprocess_data(self, data):
        data['Date'] = pd.to_datetime(data['Date'])
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Weekday'] = data['Date'].dt.weekday
        data.drop(['Date'], axis=1, inplace=True)
        return data

    def split_data(self, data):
        X = data.drop(['Profit'], axis=1)
        y = data['Profit']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred)
        return accuracy, classification

    def scale_data(self, data):
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def make_prediction(self, data):
        prediction = self.model.predict(data)
        return prediction


class Portfolio:
    def __init__(self, capital):
        self.capital = capital
        self.stocks = {}

    def buy(self, stock, quantity):
        try:
            price = yf.Ticker(stock).history(period="1d")["Close"].values[-1]
            cost = price * quantity
            if cost <= self.capital:
                self.capital -= cost
                if stock in self.stocks:
                    self.stocks[stock] += quantity
                else:
                    self.stocks[stock] = quantity
        except Exception as e:
            print(f"Error buying stock: {e}")

    def sell(self, stock, quantity):
        if stock in self.stocks and quantity <= self.stocks[stock]:
            try:
                price = yf.Ticker(stock).history(
                    period="1d")["Close"].values[-1]
                revenue = price * quantity
                self.capital += revenue
                self.stocks[stock] -= quantity
                if self.stocks[stock] == 0:
                    del self.stocks[stock]
            except Exception as e:
                print(f"Error selling stock: {e}")

    def calculate_performance(self):
        try:
            total_investment = self.capital
            current_value = self.capital
            for stock, quantity in self.stocks.items():
                price = yf.Ticker(stock).history(
                    period="1d")["Close"].values[-1]
                current_value += (price * quantity)
            performance = (current_value - total_investment) / total_investment
            return performance
        except Exception as e:
            print(f"Error calculating portfolio performance: {e}")
            return None

    def rebalance(self):
        pass


class AdaptiveLearning:
    def __init__(self):
        self.past_performances = []

    def add_performance(self, performance):
        self.past_performances.append(performance)

    def improve_model(self, model):
        try:
            X = np.arange(len(self.past_performances)).reshape(-1, 1)
            y = np.array(self.past_performances)
            model.fit(X, y)
        except Exception as e:
            print(f"Error improving model: {e}")

    def make_prediction(self, model):
        try:
            prediction = model.predict([[len(self.past_performances)]])
            return prediction[0]
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None


if __name__ == "__main__":
    # Web Content Aggregation
    aggregator = WebContentAggregator()
    aggregator.dynamic_search_query("Python programming")
    scraped_data = aggregator.web_scraping()

    # Sentiment Analysis
    sentiment_analyzer = SentimentAnalyzer()
    sentiments = []
    for content in scraped_data:
        sentiment = sentiment_analyzer.analyze_sentiment(content)
        sentiments.append(sentiment)
    aggregated_sentiment = max(set(sentiments), key=sentiments.count)

    # Keyword Extraction
    keywords = []
    for content in scraped_data:
        keywords.append(sentiment_analyzer.extract_keywords(content))

    # Content Categorization
    categorization = ContentCategorization()
    categories = ['technology', 'finance', 'health']
    categorization.train_model(scraped_data, categories)
    categorized_texts = [categorization.predict_category(
        text) for text in scraped_data]

    # Stock Prediction
    prediction = StockPrediction()
    file_name = "financial_data.csv"  # Update with the actual file name and path
    data = prediction.load_data(file_name)
    processed_data = prediction.preprocess_data(data)
    scaled_data = prediction.scale_data(processed_data)
    X_train, X_test, y_train, y_test = prediction.split_data(scaled_data)
    prediction.train_model(X_train, y_train)
    accuracy, classification = prediction.evaluate_model(X_test, y_test)

    # Portfolio Management
    portfolio = Portfolio(10000)
    investment_opportunities = [
        {"action": "buy", "stock": "AAPL", "quantity": 10},
        {"action": "sell", "stock": "TSLA", "quantity": 5},
    ]
    for opportunity in investment_opportunities:
        action = opportunity["action"]
        stock = opportunity["stock"]
        quantity = opportunity["quantity"]
        if action == "buy":
            portfolio.buy(stock, quantity)
        elif action == "sell":
            portfolio.sell(stock, quantity)

    performance = portfolio.calculate_performance()

    # Adaptive Learning
    learning = AdaptiveLearning()
    learning.add_performance(performance)
    learning.improve_model(prediction.model)
    next_prediction = learning.make_prediction(prediction.model)

    # Results
    print(aggregated_sentiment)
    print(keywords)
    print(categorized_texts)
    print(accuracy)
    print(classification)
    print(performance)
    print(next_prediction)
