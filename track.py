import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier

class ObjectTracker:
    def __init__(self, static_feature_model, full_feature_model):
        self.static_feature_model = static_feature_model
        self.full_feature_model = full_feature_model
        self.tracked_objects = pd.DataFrame()
        self.next_object_id = 1  # 用于分配新的object_id

    @staticmethod
    def calculate_euclidean_distance(self, obj1, obj2):
        return np.linalg.norm(obj1 - obj2)

    def build_cost_matrix(self, frame_n, frame_n_plus_1, w_d=1, w_c=1):
        num_objects_n = frame_n.shape[0]
        num_objects_n1 = frame_n_plus_1.shape[0]
        cost_matrix = np.zeros((num_objects_n, num_objects_n1))

        for i in range(num_objects_n):
            for j in range(num_objects_n1):
                distance = self.calculate_euclidean_distance(frame_n.iloc[i][['location_x', 'location_y', 'location_z']],
                                                             frame_n_plus_1.iloc[j][['location_x', 'location_y', 'location_z']])
                cosine_similarity = np.dot(frame_n.iloc[i]['pca_z_cosine'], frame_n_plus_1.iloc[j]['pca_z_cosine'])
                cost_matrix[i, j] = w_d * distance + w_c * (1 - cosine_similarity)

        return cost_matrix

    def update_tracking(self, df):
        df_grouped = df.groupby('SheetName')
        for sheet_name, current_frame in df_grouped:
            if sheet_name in self.tracked_objects['SheetName'].unique():
                previous_frame = self.tracked_objects[self.tracked_objects['SheetName'] == sheet_name]
                cost_matrix = self.build_cost_matrix(previous_frame, current_frame)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # 更新已追踪对象信息
                matched_indices = np.full(len(current_frame), False)
                for row, col in zip(row_ind, col_ind):
                    if cost_matrix[row, col] < 0.5:  # 使用默认阈值
                        current_frame.iloc[col, current_frame.columns.get_loc('object_id')] = previous_frame.iloc[row]['object_id']
                        matched_indices[col] = True

                # 为未匹配的对象分配新ID
                for i in np.where(~matched_indices)[0]:
                    current_frame.iloc[i, current_frame.columns.get_loc('object_id')] = self.next_object_id
                    self.next_object_id += 1

            else:
                # 为新场景的对象分配ID
                current_frame['object_id'] = np.arange(self.next_object_id, self.next_object_id + len(current_frame))
                self.next_object_id += len(current_frame)

            self.tracked_objects = pd.concat([self.tracked_objects, current_frame], ignore_index=True)

    def classify_objects(self):
        # 为追踪的对象重新分类
        for sheet_name in self.tracked_objects['SheetName'].unique():
            frame = self.tracked_objects[self.tracked_objects['SheetName'] == sheet_name]
            features = frame.drop(columns=['object_id', 'SheetName', 'timestamp'])
            predictions = self.full_feature_model.predict(features)
            self.tracked_objects.loc[self.tracked_objects['SheetName'] == sheet_name, 'full_feature_type'] = predictions

