import os
from data import PCDObjectExtractor
from feature import ObjectHandler, ObjectFeaturesExtractor, StaticFeaturesExtractor
from data import FrameDataProcessor
import pandas as pd
from random_forest import RandomForestModel
import pickle
from track import ObjectTracker

class Task:
    def __init__(self, base='data'):
        self.base_dir = base
        self.train_set, self.test_set = self.split_dataset()
        self.train_data_path, self.test_data_path = 'temp/train_features.xlsx', 'temp/test_features.xlsx'
        self.static_model_path = 'temp/static_random_forest_model.pkl'
        self.full_model_path = 'temp/full_random_forest_model.pkl'
        self.static_feature_path = 'temp/static_test_features.xlsx'

    def prepare_data(self):
        self.extract_dataset_feature(self.train_set, filename=self.train_data_path)
        self.extract_dataset_feature(self.test_set, filename=self.test_data_path)

    def split_dataset(self):
        file_path = f"{self.base_dir}/tracking_train_label"
        directories = self.list_directories(file_path)
        n = len(directories)
        train_set = directories[:4 * n // 5]
        test_set = directories[4 * n // 5:]
        return train_set, test_set

    def prepare_static_feature(self):
        features = []
        frame_ids = []
        for directory in self.test_set[:1]:
            frame_id = directory
            features.append([])
            frame_ids.append(frame_id)
            processor = PCDObjectExtractor(self.base_dir, frame_id)
            frame_data = processor.extract_objects(eps=0.5, min_points=20)
            static_extractor = StaticFeaturesExtractor(frame_data)
            features_list = static_extractor.extract_features()
            features[-1] += features_list
        self.save_data(frame_ids, features, self.static_feature_path)

    def extract_dataset_feature(self, dataset, filename):
        features = []
        frame_ids = []
        for directory in dataset:
            frame_id = directory
            features.append([])
            frame_ids.append(frame_id)
            processor = FrameDataProcessor(self.base_dir, frame_id)
            frame_data = processor.load_all_frame_data()
            targets = ObjectHandler.all_obj(frame_data)
            handler = ObjectHandler(frame_data)
            for target in targets:
                handler.object_sequence(target)
                if len(handler.obj) - handler.obj.count(None) > 1:
                    extractor = ObjectFeaturesExtractor(handler)
                    features[-1] += extractor.extract_features()
        self.save_data(frame_ids, features, filename)

    def generate_random_forest_model(self):
        train_data = self.read_all_sheets(self.train_data_path)
        test_data = self.read_all_sheets(self.test_data_path)
        static = self.model_generator(train_data, test_data, mode='static', n_estimators=50, random_state=42)
        full = self.model_generator(train_data, test_data, mode='full', n_estimators=50, random_state=42)
        self.save_model(static, self.static_model_path)
        self.save_model(full, self.full_model_path)

    def tracking(self):
        static_rf_model = self.load_model(self.static_model_path)  # 假设已经加载了训练好的模型
        full_rf_model = self.load_model(self.full_model_path)  # 假设已经加载了训练好的模型
        tracker = ObjectTracker(static_rf_model, full_rf_model)
        df = self.read_all_sheets(self.static_feature_path)

        tracker.update_tracking(df)  # 更新跟踪信息
        tracker.classify_objects()  # 重新分类物体



    @staticmethod
    def save_model(model, name):
        with open(name, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(name):
        with open(name, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    @staticmethod
    def save_data(sheet_names: list, features: list, name: str):
        # 确保页签名称列表和特征列表的长度相同
        if len(features) != len(sheet_names):
            raise ValueError("每个特征集必须有一个对应的页签名称")

        # 创建一个Excel文件写入器
        with pd.ExcelWriter(name, engine='openpyxl') as writer:
            # 遍历特征列表和页签名称列表
            for feature_list, sheet_name in zip(features, sheet_names):
                # 将子列表中的字典转换为DataFrame
                df = pd.DataFrame(feature_list)
                # 保存DataFrame到Excel的一个新页签，使用提供的页签名
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def read_sheet(filename: str, sheet_name: str) -> pd.DataFrame:
        # 从文件中读取指定的页签
        df = pd.read_excel(filename, sheet_name=sheet_name)
        return df

    @staticmethod
    def read_all_sheets(filename: str) -> pd.DataFrame:
        # 加载Excel文件中的所有页签
        xls = pd.ExcelFile(filename)
        # 创建一个空的DataFrame来存储所有页签的数据
        all_data = pd.DataFrame()
        for sheet_name in xls.sheet_names:
            # 读取每个页签的数据
            df = xls.parse(sheet_name)
            # 添加一个列来标记数据来源的页签名
            df['SheetName'] = sheet_name
            # 将这个页签的DataFrame追加到总的DataFrame中
            all_data = pd.concat([all_data, df], ignore_index=True)
        return all_data

    @staticmethod
    def list_directories(path):
        """返回指定路径下所有子文件夹的列表"""
        # 获取path下的所有条目
        entries = os.listdir(path)
        # 过滤出目录
        directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        return directories

    @staticmethod
    def model_generator(train_data, test_data, mode='static', n_estimators=50, random_state=42):
        if mode == 'static':
            features = ['length', 'width', 'height', 'volume', 'surface_area', 'xy_area', 'density', 'pca_z_cosine']
        else:
            features = ['length', 'width', 'height', 'volume', 'surface_area', 'xy_area', 'density', 'pca_z_cosine',
                        'principal_camera_cosine', 'velocity_camera_cosine', 'velocity_magnitude',
                        'acceleration_magnitude',
                        'angle_change', 'angle_speed']

        model = RandomForestModel(features, 'object_type', n_estimators=n_estimators, random_state=random_state)
        model.input_data(train_data, test_data)
        model.train()
        model.evaluate()
        return model


if __name__ == '__main__':
    task = Task()
    #task.prepare_data()
    #task.prepare_static_feature()
    #task.generate_random_forest_model()
    task.tracking()





