import os
from data import PCDObjectExtractor
from feature import ObjectHandler, ObjectFeaturesExtractor, StaticFeaturesExtractor
from data import FrameDataProcessor
import pandas as pd
from random_forest import model_generator

class Task:
    def __init__(self, base='data'):
        self.base_dir = base

    def prepare_data(self):
        train_set, test_set = self.split_dataset()

        train_features = self.extract_dataset_feature(train_set)
        test_features = self.extract_dataset_feature(test_set)

        self.save_data(train_features, 'train_features.csv')
        self.save_data(test_features, 'test_features.csv')

        static_test_features = self.process_static_feature(test_set)
        self.save_data(static_test_features, 'static_test_features.csv')
        print(static_test_features)

    def split_dataset(self):
        file_path = f"{self.base_dir}/tracking_train_label"
        directories = self.list_directories(file_path)
        n = len(directories)
        train_set = directories[:2 * n // 5]
        test_set = directories[2 * n // 5:]
        return train_set, test_set

    def process_static_feature(self, dataset):
        features = []
        for directory in dataset[:1]:
            frame_id = directory
            processor = PCDObjectExtractor(self.base_dir, frame_id)
            frame_data = processor.extract_objects(eps=0.5, min_points=20)
            static_extractor = StaticFeaturesExtractor(frame_data)
            features_list = static_extractor.extract_features()
            features.append(features_list)
        return features

    def extract_dataset_feature(self, dataset):
        features = []
        for directory in dataset:
            frame_id = directory
            processor = FrameDataProcessor(self.base_dir, frame_id)
            frame_data = processor.load_all_frame_data()
            targets = ObjectHandler.all_obj(frame_data)
            handler = ObjectHandler(frame_data)
            for target in targets:
                handler.object_sequence(target)
                if len(handler.obj) - handler.obj.count(None) > 1:
                    extractor = ObjectFeaturesExtractor(handler)
                    features += extractor.extract_features()
        return features

    def predict_label(self, feature_file):
        df = self.load_data(feature_file)
        static = model_generator(mode='static')
        full = model_generator(mode='full')
        label = static.predict(df)
        print(label)

    @staticmethod
    def save_data(features: list, name: str):
        # 将特征列表转换为DataFrame
        df = pd.DataFrame(features)
        # 保存DataFrame到CSV文件
        df.to_csv(name, index=False)

    @staticmethod
    def load_data(filename: str):
        """从CSV文件加载数据并返回DataFrame"""
        # 使用pandas的read_csv函数读取CSV文件
        df = pd.read_csv(filename)
        return df

    @staticmethod
    def list_directories(path):
        """返回指定路径下所有子文件夹的列表"""
        # 获取path下的所有条目
        entries = os.listdir(path)
        # 过滤出目录
        directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        return directories


if __name__ == '__main__':
    task = Task()
    task.prepare_data()
    task.predict_label(feature_file='static_test_features.csv')





