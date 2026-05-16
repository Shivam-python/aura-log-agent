import joblib
from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class LogClassifier:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

        # LOAD trained model
        self.clf = joblib.load(
            "models/log_classifier.joblib"
        )

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        report = classification_report(y_test, predictions)

        print(report)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def predict(self, log_message):
        embedding = self.embedding_model.encode(
            [log_message]
        )

        prediction = self.clf.predict(embedding)

        return prediction[0]