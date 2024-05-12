import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, features, target, n_estimators=100, random_state=42):
        self.features = features
        self.target = target
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self, train_path, test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

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


def model_generator(mode='static', n_estimators=100, random_state=42):
    if mode == 'static':
        features = ['length', 'width', 'height', 'volume', 'surface_area', 'xy_area', 'density', 'pca_z_cosine']
    else:
        features = ['length', 'width', 'height', 'volume', 'surface_area', 'xy_area', 'density', 'pca_z_cosine',
                    'principal_camera_cosine', 'velocity_camera_cosine', 'velocity_magnitude', 'acceleration_magnitude',
                    'angle_change', 'angle_speed']

    model = RandomForestModel(features, 'object_type', n_estimators=n_estimators, random_state=random_state)
    model.load_data('train_features.csv', 'test_features.csv')
    model.train()
    #model.evaluate()
    return model


if __name__ == "__main__":
    static = model_generator(mode='static')
    full = model_generator(mode='full')
