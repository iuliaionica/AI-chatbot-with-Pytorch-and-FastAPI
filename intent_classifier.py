import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class IntentClassifier:
    def __init__(self, path="data/intents.csv"):
        self.data = pd.read_csv(path)  # Load labeled samples
        self.vectorizer = TfidfVectorizer()  # Convert text to numbers
        self.model = MultinomialNB()  # Naive Bayes classifier
        self._train()

    def _train(self):
        X = self.vectorizer.fit_transform(self.data['text'])
        y = self.data['intent']
        self.model.fit(X, y)  # Train classifier

    def predict_intent(self, text: str) -> str:
        X_test = self.vectorizer.transform([text])
        return self.model.predict(X_test)[0]
