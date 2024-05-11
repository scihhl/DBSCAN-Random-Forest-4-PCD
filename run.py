import os
import pickle
from feature import ObjectHandler, ObjectFeaturesExtractor
from data import FrameDataProcessor
import pandas as pd

class Task:
    def __init__(self, base='data'):
        self.base_dir = base

    def prepare_data(self):
        train_set, test_set = self.split_dataset()

        train_features = self.extract_dataset_feature(train_set)
        test_features = self.extract_dataset_feature(train_set)

        self.save_data(train_features, 'train_features.csv')
        self.save_data(test_features, 'train_features.csv')

    def split_dataset(self):
        file_path = f"{self.base_dir}/tracking_train_label"
        directories = self.list_directories(file_path)
        n = len(directories)
        train_set = directories[:2 * n // 3]
        test_set = directories[2 * n // 3:]
        return train_set, test_set

    def extract_dataset_feature(self, train_set):
        features = []
        for directory in train_set:
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





