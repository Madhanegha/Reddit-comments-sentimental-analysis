from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

app = Flask(__name__)

# Load the sentiment analysis model
sentiment_model = joblib.load('model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        comment = request.form['comment']
        result = sentiment_model.polarity_scores(comment)

        if result['pos'] >= 0.5:
            sentiment = "Positive"
        elif result['neu'] >= 0.5:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"

        return render_template('result.html', comment=comment, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)