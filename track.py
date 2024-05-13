import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ObjectTracker:
    def __init__(self, static_feature_model, full_feature_model):
        self.static_feature_model = static_feature_model
        self.full_feature_model = full_feature_model
        self.tracked_objects = pd.DataFrame(columns=['SheetName', 'object_id'])
        self.next_object_id = 1  # 在构造函数中正确初始化 object_id 计数器
        self.static_feature = ['length', 'width', 'height', 'volume', 'surface_area', 'xy_area', 'density',
                               'pca_z_cosine']
        self.full_feature = ['length', 'width', 'height', 'volume', 'surface_area', 'xy_area', 'density',
                             'pca_z_cosine',
                             'principal_camera_cosine', 'velocity_camera_cosine', 'velocity_magnitude',
                             'acceleration_magnitude',
                             'angle_change', 'angle_speed']

    @staticmethod
    def calculate_euclidean_distance(obj1, obj2):
        return np.linalg.norm(obj1 - obj2)

    def predict_labels(self, frame):
        # 使用static_feature_model预测每个对象的标签
        return self.static_feature_model.predict(frame)

    @staticmethod
    def standardize_features(dataframe, model_features):
        # 仅对模型特征进行标准化
        dataframe[model_features] = StandardScaler().fit_transform(dataframe[model_features])
        return dataframe

    @staticmethod
    def get_feature_weights(rf_model):
        feature_weights = rf_model.model.feature_importances_
        total_importance = sum(feature_weights)
        normalized_feature_weights = feature_weights / total_importance
        return normalized_feature_weights

    @staticmethod
    def calculate_weighted_cosine_similarity(vec1, vec2, weights):
        weighted_vec1 = vec1 * weights
        weighted_vec2 = vec2 * weights
        dot_product = np.dot(weighted_vec1, weighted_vec2)
        norm1 = np.linalg.norm(weighted_vec1)
        norm2 = np.linalg.norm(weighted_vec2)
        return dot_product / (norm1 * norm2) if (norm1 != 0 and norm2 != 0) else 0

    def build_cost_matrix(self, frame_n, frame_n_plus_1):
        model_features = self.static_feature

        # 获取权重
        weights = self.get_feature_weights(self.static_feature_model)

        # 重置索引以确保索引连续
        frame_n = frame_n.reset_index(drop=True)
        frame_n_plus_1 = frame_n_plus_1.reset_index(drop=True)

        # 标准化特征
        frame_n = self.standardize_features(frame_n, model_features)
        frame_n_plus_1 = self.standardize_features(frame_n_plus_1, model_features)

        # 预测标签
        frame_n['label'] = self.static_feature_model.predict(frame_n[model_features])
        frame_n_plus_1['label'] = self.static_feature_model.predict(frame_n_plus_1[model_features])

        # 过滤数据，仅保留关注的类别，例如标签不是6的类
        frame_n = frame_n[frame_n['label'] != 6].reset_index(drop=True)
        frame_n_plus_1 = frame_n_plus_1[frame_n_plus_1['label'] != 6].reset_index(drop=True)

        num_objects_n = frame_n.shape[0]
        num_objects_n1 = frame_n_plus_1.shape[0]
        cost_matrix = np.zeros((num_objects_n, num_objects_n1))

        for i in range(num_objects_n):
            for j in range(num_objects_n1):
                distance = self.calculate_euclidean_distance(frame_n.iloc[i][model_features],
                                                             frame_n_plus_1.iloc[j][model_features])
                cosine_similarity = self.calculate_weighted_cosine_similarity(
                    frame_n.iloc[i][model_features].values, frame_n_plus_1.iloc[j][model_features].values, weights
                )
                label_mismatch_penalty = 20 if frame_n.iloc[i]['label'] != frame_n_plus_1.iloc[j]['label'] else 0

                cost_matrix[i, j] = distance + (1 - cosine_similarity) * 20 + label_mismatch_penalty

        return cost_matrix

    def preprocess_data(self, df):
        model_features = self.static_feature
        # 预测标签
        df['label'] = self.static_feature_model.predict(df[model_features])
        # 剔除标签为6的数据
        df = df[df['label'] != 6]
        return df

    def update_tracking(self, df):
        # 初步预测并剔除不相关的标签
        df = self.preprocess_data(df)

        df_sorted = df.sort_values(['SheetName', 'timestamp'])
        grouped_by_sheet = df_sorted.groupby('SheetName')

        all_tracked_frames = []

        for sheet_name, sheet_data in grouped_by_sheet:
            previous_frame = None
            for timestamp, current_frame in sheet_data.groupby('timestamp'):
                current_frame = current_frame.reset_index(drop=True)  # 重置索引以保证连续性

                if previous_frame is not None:
                    cost_matrix = self.build_cost_matrix(previous_frame, current_frame)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    matched_indices = np.full(len(current_frame), False)
                    for row, col in zip(row_ind, col_ind):
                        if cost_matrix[row, col] < 20:  # 使用阈值
                            current_frame.loc[col, 'object_id'] = previous_frame.loc[row, 'object_id']
                            matched_indices[col] = True

                    for i in np.where(~matched_indices)[0]:
                        current_frame.loc[i, 'object_id'] = self.next_object_id
                        self.next_object_id += 1
                else:
                    current_frame['object_id'] = np.arange(self.next_object_id,
                                                           self.next_object_id + len(current_frame))
                    self.next_object_id += len(current_frame)

                previous_frame = current_frame.copy()
                all_tracked_frames.append(current_frame)  # 将处理过的帧加入列表

        # 更新追踪对象，一次性合并所有处理过的帧
        self.tracked_objects = pd.concat(all_tracked_frames, ignore_index=True)
    def classify_objects(self):
        if not self.tracked_objects.empty:
            for sheet_name in self.tracked_objects['SheetName'].unique():
                frame = self.tracked_objects[self.tracked_objects['SheetName'] == sheet_name]
                features = frame.drop(columns=['object_id', 'SheetName', 'timestamp'])
                predictions = self.full_feature_model.predict(features)
                self.tracked_objects.loc[
                    self.tracked_objects['SheetName'] == sheet_name, 'full_feature_type'] = predictions

    def refine_classification_with_motion(self):
        # 假设所有需要的运动特征和时间戳都已经计算并存储在self.tracked_objects中
        if not self.tracked_objects.empty:
            # 计算每个对象的运动特征
            for object_id in self.tracked_objects['object_id'].unique():
                obj_data = self.tracked_objects[self.tracked_objects['object_id'] == object_id]
                location_vectors = obj_data[['location_x', 'location_y', 'location_z']].values
                time_vectors = obj_data['timestamp'].values

                # 使用 ObjectFeaturesExtractor 计算运动特征
                movement, velocity, acceleration, _, _ = ObjectFeaturesExtractor.compute_movement(location_vectors,
                                                                                                  time_vectors)

                # 可以在这里添加这些特征到DataFrame
                self.tracked_objects.loc[self.tracked_objects['object_id'] == object_id, 'velocity'] = velocity
                self.tracked_objects.loc[self.tracked_objects['object_id'] == object_id, 'acceleration'] = acceleration

            # 假设有一个方法重新加载运动特征到模型并预测
            self.reclassify_with_motion_features()
