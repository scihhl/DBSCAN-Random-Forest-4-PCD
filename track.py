import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from feature import ObjectFeaturesExtractor


class ObjectTracker:
    def __init__(self):
        self.tracked_objects = pd.DataFrame(columns=['SheetName', 'object_id'])
        self.next_object_id = 1  # 在构造函数中正确初始化 object_id 计数器
        self.object_appearance_count = {}
        self.all_tracked_frames = []
        self.tools = ObjectFeaturesExtractor

    @staticmethod
    def calculate_euclidean_distance(obj1, obj2):
        return np.linalg.norm(obj1 - obj2)

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

    def build_cost_matrix(self, frame_n, frame_n_plus_1, random_forest_model, model_features):

        # 获取权重
        weights = self.get_feature_weights(random_forest_model)

        # 重置索引以确保索引连续
        frame_n = frame_n.reset_index(drop=True)
        frame_n_plus_1 = frame_n_plus_1.reset_index(drop=True)

        # 标准化特征
        frame_n = self.standardize_features(frame_n, model_features)
        frame_n_plus_1 = self.standardize_features(frame_n_plus_1, model_features)

        # 预测标签
        frame_n['label'] = random_forest_model.predict(frame_n[model_features])
        frame_n_plus_1['label'] = random_forest_model.predict(frame_n_plus_1[model_features])

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

    @staticmethod
    def preprocess_data(df, model, model_features):
        # 预测标签
        df['label'] = model.predict(df[model_features])
        # 剔除标签为6的数据
        df = df[df['label'] != 6]
        return df

    def update_tracking(self, df, random_forest_model, feature):

        df_sorted = df.sort_values(['SheetName', 'timestamp'])
        grouped_by_sheet = df_sorted.groupby('SheetName')

        for sheet_name, sheet_data in grouped_by_sheet:
            previous_frame = None
            for timestamp, current_frame in sheet_data.groupby('timestamp'):
                current_frame = current_frame.reset_index(drop=True)  # 重置索引以保证连续性

                if previous_frame is not None:
                    cost_matrix = self.build_cost_matrix(previous_frame, current_frame, random_forest_model, feature)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    matched_indices = np.full(len(current_frame), False)
                    for row, col in zip(row_ind, col_ind):
                        if cost_matrix[row, col] < 20:  # 使用阈值
                            obj_id = previous_frame.loc[row, 'object_id']
                            current_frame.loc[col, 'object_id'] = obj_id
                            matched_indices[col] = True
                            self.object_appearance_count[obj_id] = self.object_appearance_count.get(obj_id, 0) + 1

                    for i in np.where(~matched_indices)[0]:
                        current_frame.loc[i, 'object_id'] = self.next_object_id
                        self.object_appearance_count[self.next_object_id] = 1
                        self.next_object_id += 1
                else:
                    # 为第一帧中的每个对象初始化一个新的ID
                    for i in range(len(current_frame)):
                        current_frame.loc[i, 'object_id'] = self.next_object_id
                        self.object_appearance_count[self.next_object_id] = 1
                        self.next_object_id += 1

                previous_frame = current_frame.copy()
                self.all_tracked_frames.append(current_frame)

        # 结束追踪后剔除未达到出现阈值的物体
        self.tracked_objects = pd.concat(self.all_tracked_frames, ignore_index=True)
        valid_ids = {k for k, v in self.object_appearance_count.items() if v >= 2}
        self.tracked_objects = self.tracked_objects[self.tracked_objects['object_id'].isin(valid_ids)]
        return self.tracked_objects

    def compute_motion_features(self):
        grouped_data = self.tracked_objects.groupby('object_id')
        self.tracked_objects['principal_camera_cosine'] = np.nan
        self.tracked_objects['velocity_camera_cosine'] = np.nan
        self.tracked_objects['velocity_magnitude'] = np.nan
        self.tracked_objects['acceleration_magnitude'] = np.nan
        self.tracked_objects['angle_change'] = np.nan
        self.tracked_objects['angle_speed'] = np.nan

        for object_id, group in grouped_data:
            if len(group) > 1:  # 确保有足够的数据点来计算运动特征
                timestamps = group['timestamp'].to_numpy()
                locations = group[['location_x', 'location_y', 'location_z']].to_numpy()
                camera_locations = group[['camera_location_x', 'camera_location_y', 'camera_location_z']].to_numpy()
                principal_axis = group[['principal_axis_x', 'principal_axis_y', 'principal_axis_z']].to_numpy()

                (movements, velocities, accelerations,
                 angle_changes, angle_speeds) = self.tools.compute_movement(locations, timestamps)

                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

                (camera_movements, camera_velocities, camera_accelerations,
                 camera_angle_changes, camera_angle_change_speeds) = (
                    self.tools.compute_movement(camera_locations, timestamps))

                idx = group.index
                for i in range(len(group)):
                    self.tracked_objects.at[idx[i], 'principal_camera_cosine'] = \
                        abs(self.tools.compute_cosine_similarity(principal_axis[i], camera_velocities[i]))
                    self.tracked_objects.at[idx[i], 'velocity_camera_cosine'] = \
                        abs(self.tools.compute_cosine_similarity(velocities[i], camera_velocities[i]))
                    self.tracked_objects.at[idx[i], 'velocity_magnitude'] = velocity_magnitudes[i]
                    self.tracked_objects.at[idx[i], 'acceleration_magnitude'] = acceleration_magnitudes[i]
                    self.tracked_objects.at[idx[i], 'angle_change'] = angle_changes[i]
                    self.tracked_objects.at[idx[i], 'angle_speed'] = angle_speeds[i]
        return self.tracked_objects
