import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier

class ObjectTracker:
    def __init__(self, static_feature_model, full_feature_model):
        """
        初始化追踪器
        :param static_feature_model: 预训练的静态特征随机森林模型
        :param full_feature_model: 预训练的全特征随机森林模型
        """
        self.static_feature_model = static_feature_model
        self.full_feature_model = full_feature_model
        self.tracked_objects = {}
        self.next_object_id = 1  # 用于分配新的object_id

    @staticmethod
    def calculate_euclidean_distance(obj1, obj2):
        return np.linalg.norm(np.array(obj1['position']) - np.array(obj2['position']))

    @staticmethod
    def calculate_cosine_similarity(features1, features2):
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        return dot_product / (norm1 * norm2)

    def build_cost_matrix(self, objects_frame_n, objects_frame_n_plus_1, w_d=1, w_c=1):
        num_objects_n = len(objects_frame_n)
        num_objects_n1 = len(objects_frame_n_plus_1)
        cost_matrix = np.zeros((num_objects_n, num_objects_n1))

        for i, obj_n in enumerate(objects_frame_n):
            for j, obj_n1 in enumerate(objects_frame_n_plus_1):
                distance = self.calculate_euclidean_distance(obj_n, obj_n1)
                cosine_similarity = self.calculate_cosine_similarity(obj_n['features'], obj_n1['features'])
                cost_matrix[i, j] = w_d * distance + w_c * (1 - cosine_similarity)

        return cost_matrix

    def update_tracking(self, frames, cost_threshold=0.5):
        """
        更新物体追踪状态
        :param frames: 包含连续帧中所有物体静态特征的列表
        :param cost_threshold: 成本函数阈值
        """
        previous_frame = frames[0]
        for current_frame in frames[1:]:
            cost_matrix = self.build_cost_matrix(previous_frame, current_frame)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            new_tracked_objects = {}
            for row, col in zip(row_ind, col_ind):
                if cost_matrix[row, col] < cost_threshold:
                    obj_id = previous_frame[row].get('object_id', None)
                    if obj_id is None:
                        obj_id = self.next_object_id
                        self.next_object_id += 1
                    new_tracked_objects[obj_id] = current_frame[col]
                else:
                    # 如果没有匹配，分配新的ID
                    new_tracked_objects[self.next_object_id] = current_frame[col]
                    self.next_object_id += 1

            self.tracked_objects.update(new_tracked_objects)
            previous_frame = current_frame

    def classify_objects(self):
        """
        使用全特征模型对追踪到的物体重新分类
        """
        for obj_id, obj in self.tracked_objects.items():
            features = np.array([obj['features']])
            prediction = self.full_feature_model.predict(features)
            obj['full_feature_type'] = prediction[0]

if __name__ == '__main__':
    # 使用示例
    static_rf_model = RandomForestClassifier()  # 假设已经加载了训练好的模型
    full_rf_model = RandomForestClassifier()    # 假设已经加载了训练好的模型
    tracker = ObjectTracker(static_rf_model, full_rf_model)
    initial_frames = [ ... ]  # 初始帧数据
    tracker.update_tracking(initial_frames)  # 更新跟踪信息
    tracker.classify_objects()  # 重新分类物体
