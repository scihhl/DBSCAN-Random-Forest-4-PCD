import numpy as np
from scipy.spatial.distance import euclidean, cosine


class EvaluationMetrics:
    def __init__(self, pred, true, iou_threshold=0.5):
        self.pred = pred
        self.true = true
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_iou(bbox_pred, bbox_true):
        x1 = max(bbox_pred[0], bbox_true[0])
        y1 = max(bbox_pred[1], bbox_true[1])
        x2 = min(bbox_pred[0] + bbox_pred[2], bbox_true[0] + bbox_true[2])
        y2 = min(bbox_pred[1] + bbox_pred[3], bbox_true[1] + bbox_true[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        pred_area = bbox_pred[2] * bbox_pred[3]
        true_area = bbox_true[2] * bbox_true[3]
        union_area = pred_area + true_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    @staticmethod
    def calculate_position_error(pred_pos, true_pos):
        return euclidean(pred_pos, true_pos)

    @staticmethod
    def calculate_category_accuracy(pred_cat, true_cat):
        return 1 if pred_cat == true_cat else 0

    @staticmethod
    def calculate_feature_similarity(pred_vec, true_vec):
        return 1 - cosine(pred_vec, true_vec)

    def find_closest_object(self, pred_obj, true_objects):
        min_distance = float('inf')
        closest_obj = None
        for obj in true_objects:
            dist = euclidean(pred_obj['position'], obj['position'])
            if dist < min_distance:
                min_distance = dist
                closest_obj = obj
        return closest_obj

    def evaluate(self):
        results = {'average_position_error': [], 'category_accuracy': [], 'average_iou': [],
                   'feature_vector_similarity': []}
        for timestamp_pred, timestamp_true in zip(self.pred, self.true):
            for pred_obj in timestamp_pred:
                closest_obj = self.find_closest_object(pred_obj, timestamp_true)

                # 计算位置误差
                pos_error = self.calculate_position_error(pred_obj['position'], closest_obj['position'])
                results['average_position_error'].append(pos_error)

                # 计算类别准确性
                cat_accuracy = self.calculate_category_accuracy(pred_obj['category'], closest_obj['category'])
                results['category_accuracy'].append(cat_accuracy)

                # 计算IoU
                iou = self.calculate_iou(pred_obj.get('bbox', []), closest_obj.get('bbox', []))
                results['average_iou'].append(iou)

                # 计算特征向量相似度
                similarity = self.calculate_feature_similarity(pred_obj['full_feature_vector'],
                                                               closest_obj['full_feature_vector'])
                results['feature_vector_similarity'].append(similarity)

        # 计算平均值
        metrics = {
            'Mean Position Error': np.mean(results['average_position_error']),
            'Category Accuracy': np.mean(results['category_accuracy']),
            'Mean IoU': np.mean(results['average_iou']),
            'Mean Feature Vector Similarity': np.mean(results['feature_vector_similarity'])
        }
        return metrics