import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class EDA:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        # Explicitly specify the columns that are features
        self.feature_columns = ['length', 'width', 'height', 'volume', 'surface_area',
                                'xy_area', 'density', 'pca_z_cosine', 'principal_camera_cosine',
                                'velocity_camera_cosine', 'velocity_magnitude', 'acceleration_magnitude', 'angle_change', 'angle_speed']
        self.features = self.data[self.feature_columns]
        self.target = self.data['object_type']
        self.plots_dir = 'temp/'

    def plot_histograms(self):
        self.features.hist(bins=15, figsize=(15, 10))
        plt.savefig(self.plots_dir + 'histograms.png')
        plt.close()

    def plot_box_plots(self):
        for col in self.features.columns:
            sns.boxplot(x=self.target, y=self.features[col])
            plt.title(f'Box Plot of {col}')
            plt.savefig(self.plots_dir + f'box_plot_{col}.png')
            plt.close()

    def plot_scatter_plots(self):
        sns.pairplot(self.data, hue='object_type', vars=self.feature_columns)
        plt.savefig(self.plots_dir + 'scatter_plots.png')
        plt.close()

    def correlation_matrix(self):
        corr_matrix = self.features.corr()
        plt.figure(figsize=(24, 20))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(self.plots_dir + 'correlation_matrix.png')
        plt.close()

    def feature_importance(self):
        rf = RandomForestClassifier()
        rf.fit(self.features, self.target)
        importance = rf.feature_importances_
        indices = np.argsort(importance)[::-1]
        plt.figure(figsize=(10, 8))
        sns.barplot(x=self.features.columns[indices], y=importance[indices])
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.savefig(self.plots_dir + 'feature_importance.png')
        plt.close()

    def run_analysis(self):
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        self.plot_histograms()
        self.plot_box_plots()
        self.plot_scatter_plots()
        self.correlation_matrix()
        self.feature_importance()

if __name__ == "__main__":
    eda = EDA('temp/train_features.xlsx')
    eda.run_analysis()
