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
        self.tracked_objects = {}  # 存储跟踪的物体及其历史

    def compute_cost_matrix(self, objects_frame_n, objects_frame_n_plus_1):
        """
        计算成本矩阵，基于物体的静态特征
        """
        num_objects_n = len(objects_frame_n)
        num_objects_n1 = len(objects_frame_n_plus_1)
        cost_matrix = np.zeros((num_objects_n, num_objects_n1))

        for i, obj_n in enumerate(objects_frame_n):
            for j, obj_n1 in enumerate(objects_frame_n_plus_1):
                # 简单使用欧几里得距离计算成本
                cost = np.linalg.norm(np.array([obj_n['length'], obj_n['width'], obj_n['height']]) -
                                      np.array([obj_n1['length'], obj_n1['width'], obj_n1['height']]))
                cost_matrix[i, j] = cost

        return cost_matrix

    def update_tracking(self, frames):
        """
        更新物体追踪状态
        :param frames: 包含连续帧中所有物体静态特征的列表
        """
        for i in range(len(frames) - 1):
            cost_matrix = self.compute_cost_matrix(frames[i], frames[i + 1])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 使用匈牙利算法结果更新跟踪信息
            for row, col in zip(row_ind, col_ind):
                if cost_matrix[row, col] < some_threshold:  # 设置一个阈值来决定是否是同一个物体
                    # 更新或创建跟踪对象
                    object_id = frames[i][row].get('object_id', None)
                    if object_id:
                        self.tracked_objects[object_id] = frames[i + 1][col]
                    else:
                        new_id = max(self.tracked_objects.keys(), default=0) + 1
                        self.tracked_objects[new_id] = frames[i + 1][col]

    def classify_objects(self):
        """
        使用全特征模型对追踪到的物体重新分类
        """
        for obj_id, features in self.tracked_objects.items():
            # 假设我们可以直接从features字典获取模型需要的全部特征
            full_features = np.array([features[f] for f in self.full_feature_model.feature_importances_]).reshape(1, -1)
            prediction = self.full_feature_model.predict(full_features)
            self.tracked_objects[obj_id]['full_feature_type'] = prediction[0]

if __name__ == '__main__':

    # 使用示例
    static_rf_model = RandomForestClassifier()  # 假设已经加载了训练好的模型
    full_rf_model = RandomForestClassifier()    # 假设已经加载了训练好的模型
    tracker = ObjectTracker(static_rf_model, full_rf_model)
