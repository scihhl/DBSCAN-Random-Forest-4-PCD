from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, features, target, n_estimators=100, random_state=42):
        self.features = features
        self.target = target
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def input_data(self, train_data, test_data):
        self.X_train = train_data[self.features]
        self.y_train = train_data[self.target]
        self.X_test = test_data[self.features]
        self.y_test = test_data[self.target]

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"Accuracy: {accuracy}")

    def predict(self, X):
        X = X[self.features]
        return self.model.predict(X)


