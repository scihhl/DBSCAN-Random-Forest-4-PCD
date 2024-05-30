import numpy as np
import trimesh
from scipy.spatial.distance import euclidean
from pedestration_identify import PredestinationTracker


class EvaluationMetrics:
    def __init__(self, df_pred, df_true, iou_threshold=0.1):
        self.df_pred = df_pred.sort_values(by='timestamp').reset_index(drop=True)
        self.df_true = df_true.sort_values(by='timestamp').reset_index(drop=True)
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_iou(bbox_pred, bbox_true):
        """Calculate the IoU of two oriented bounding boxes using trimesh."""
        # Calculate the intersection volume
        intersection_mesh = bbox_pred.intersection(bbox_true)
        intersection_volume = intersection_mesh.volume if intersection_mesh.is_volume else 0

        # Calculate the union volume
        union_volume = bbox_pred.volume + bbox_true.volume - intersection_volume

        # Calculate the IoU
        iou = intersection_volume / union_volume if union_volume > 0 else 0
        return iou

    @staticmethod
    def create_oriented_bbox(center, dimensions, angles):
        """Create an oriented bounding box using trimesh."""
        obb = trimesh.primitives.Box(extents=dimensions,
                                     transform=trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2]))
        obb.apply_translation(center)
        return obb

    def evaluate(self):
        grouped_pred = self.df_pred.groupby('timestamp')
        grouped_true = self.df_true.groupby('timestamp')
        results = {'position_error': [], 'iou': []}
        matched_true_objects = set()

        timestamps = set(grouped_pred.groups.keys()) & set(grouped_true.groups.keys())

        for timestamp in timestamps:
            df_pred_timestamp = grouped_pred.get_group(timestamp)
            df_true_timestamp = grouped_true.get_group(timestamp)
            n = df_pred_timestamp.iloc[0].name
            closest_indices, min_distances = self.find_closest_matches(df_pred_timestamp, df_true_timestamp)

            for idx, pred_obj in df_pred_timestamp.iterrows():
                if pred_obj['length'] > 10 or pred_obj['width'] > 5 or pred_obj['height'] > 5:
                    continue
                closest_idx = closest_indices[idx-n]
                closest_obj = df_true_timestamp.iloc[closest_idx]
                pred_bbox = self.create_oriented_bbox(pred_obj[['location_x', 'location_y', 'location_z']].values,
                                                      pred_obj[['length', 'width', 'height']].values,
                                                      pred_obj[['principal_axis_x', 'principal_axis_y',
                                                                'principal_axis_z']].values)
                true_bbox = self.create_oriented_bbox(closest_obj[['location_x', 'location_y', 'location_z']].values,
                                                      closest_obj[['length', 'width', 'height']].values,
                                                      closest_obj[['principal_axis_x', 'principal_axis_y',
                                                                   'principal_axis_z']].values)

                iou = self.calculate_iou(pred_bbox, true_bbox)
                pos_error = euclidean(pred_obj[['location_x', 'location_y', 'location_z']].values,
                                      closest_obj[['location_x', 'location_y', 'location_z']].values)
                results['iou'].append(iou)
                if iou >= self.iou_threshold:
                    results['position_error'].append(pos_error)
                    matched_true_objects.add(closest_obj.name)

        metrics = {
            'Mean Position Error': np.mean(results['position_error']),
            'Mean IoU': np.mean(results['iou']),
            'Precision': len(matched_true_objects) / self.df_pred['object_id'].nunique() if self.df_pred['object_id'].nunique() > 0 else 0,
            'Recall': len(matched_true_objects) / self.df_true['object_id'].size if self.df_true['object_id'].size > 0 else 0
        }
        return metrics

    @staticmethod
    def find_closest_matches(df_pred, df_true):
        distances = PredestinationTracker.calculate_euclidean_distance(df_pred, df_true)
        closest_indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        return closest_indices, min_distances
