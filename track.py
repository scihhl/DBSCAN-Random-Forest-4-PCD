import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier

class ObjectTracker:
    def __init__(self, static_feature_model, full_feature_model, initial_frames):
        """
        初始化追踪器
        :param static_feature_model: 预训练的静态特征随机森林模型
        :param full_feature_model: 预训练的全特征随机森林模型
        :param initial_frames: 初始帧的数据，包含物体的时间戳、位置信息和静态物理量
        """
        self.static_feature_model = static_feature_model
        self.full_feature_model = full_feature_model
        self.tracked_objects = {}  # 存储跟踪的物体及其历史
        self.frames = initial_frames  # 初始化场景的物体识别信息

    @staticmethod
    def calculate_euclidean_distance(obj1, obj2):
        return np.linalg.norm(np.array(obj1['position']) - np.array(obj2['position']))

    @staticmethod
    def calculate_cosine_similarity(features1, features2):
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        return dot_product / (norm1 * norm2)

    @staticmethod
    def build_cost_matrix(objects_frame_n, objects_frame_n_plus_1, w_d=1, w_c=1):
        num_objects_n = len(objects_frame_n)
        num_objects_n1 = len(objects_frame_n_plus_1)
        cost_matrix = np.zeros((num_objects_n, num_objects_n1))

        for i, obj_n in enumerate(objects_frame_n):
            for j, obj_n1 in enumerate(objects_frame_n_plus_1):
                distance = ObjectTracker.calculate_euclidean_distance(obj_n, obj_n1)
                cosine_similarity = ObjectTracker.calculate_cosine_similarity(obj_n['features'], obj_n1['features'])
                cost_matrix[i, j] = w_d * distance + w_c * (1 - cosine_similarity)

        return cost_matrix

    def update_tracking(self, cost_threshold=0.5):
        """
        使用场景数据更新物体追踪状态
        :param cost_threshold: 成本函数阈值
        """
        for i in range(len(self.frames) - 1):
            cost_matrix = self.build_cost_matrix(self.frames[i], self.frames[i + 1])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 使用匈牙利算法结果更新跟踪信息
            for row, col in zip(row_ind, col_ind):
                if cost_matrix[row, col] < cost_threshold:
                    # 更新或创建跟踪对象
                    object_id = self.frames[i][row].get('object_id', None)
                    if object_id:
                        self.tracked_objects[object_id] = self.frames[i + 1][col]
                    else:
                        new_id = max(self.tracked_objects.keys(), default=0) + 1
                        self.tracked_objects[new_id] = self.frames[i + 1][col]

    def classify_objects(self):
        """
        使用全特征模型对追踪到的物体重新分类
        """
        for obj_id, features in self.tracked_objects.items():
            # 假设可以直接从features字典获取模型需要的全部特征
            full_features = np.array([features[f] for f in self.full_feature_model.feature_importances_]).reshape(1, -1)
            prediction = self.full_feature_model.predict(full_features)
            self.tracked_objects[obj_id]['full_feature_type'] = prediction[0]

if __name__ == '__main__':
    # 示例使用
    static_rf_model = RandomForestClassifier()  # 预训练的模型
    full_rf_model = RandomForestClassifier()    # 预训练的模型
    initial_data = [ ... ]  # 初始化帧数据
    tracker = ObjectTracker(static_rf_model, full_rf_model, initial_data)
    tracker.update_tracking()  # 更新跟踪信息
    tracker.classify_objects()  # 重新分类物体
