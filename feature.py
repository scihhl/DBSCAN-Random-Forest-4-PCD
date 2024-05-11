import numpy as np
import open3d as o3d
from data import FrameDataProcessor
from data import visualization
from PCA_executor import PCAExecutor
class ObjectFeaturesExtractor:
    def __init__(self, labels, poses, timestamps):
        """
        初始化特征提取器
        :param labels: 一个列表，包含连续帧中的点云对象
        :param poses: 一个列表，包含连续帧中的摄像机信息
        :param timestamps: 与点云相对应的时间戳列表
        """
        self.point_clouds = labels
        self.poses = poses
        self.timestamps = timestamps


    @staticmethod
    def compute_principal_axes(point_cloud):
        """
        计算并返回点云的主成分（PCA）方向
        """
        points_matrix = np.asarray(point_cloud.points)
        mean = np.mean(points_matrix, axis=0)
        cov_matrix = np.cov((points_matrix - mean).T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # 根据特征值排序，返回特征向量
        sort_indices = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, sort_indices]

    @staticmethod
    def compute_bounding_box(point_cloud):
        """
        计算点云的边界框尺寸
        """
        obb = point_cloud.get_oriented_bounding_box()
        return obb.extent

    def compute_density(self, point_cloud):
        """
        计算点云的密度
        """
        volume = np.prod(self.compute_bounding_box(point_cloud))
        return len(point_cloud) / volume if volume != 0 else 0

    @staticmethod
    def compute_diff(vector):
        diff_vectors = np.diff(vector, axis=0)
        # 创建一个新的数组，第一个位置用第一个差分向量填充
        extended_diff_vectors = np.empty_like(vector)
        extended_diff_vectors[1:] = diff_vectors
        extended_diff_vectors[0] = diff_vectors[0]  # 假设第一个差分与第一个计算的差分相同
        return extended_diff_vectors

    @staticmethod
    def compute_timediff(timestamps):
        time_intervals = np.diff(timestamps)
        # 补充第一个时间间隔，假设第一个间隔与后面的间隔相同
        extended_time_intervals = np.insert(time_intervals, 0, time_intervals[0])
        return extended_time_intervals

    @staticmethod
    def compute_velocity(position_vectors, extended_time_intervals):
        position_diff_vectors = ObjectFeaturesExtractor.compute_diff(position_vectors)
        return position_diff_vectors / extended_time_intervals

    @staticmethod
    def compute_acceleration(velocity_vectors, extended_time_intervals):
        velocity_diff_vectors = ObjectFeaturesExtractor.compute_diff(velocity_vectors)
        return velocity_diff_vectors / extended_time_intervals

    @staticmethod
    def compute_cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norms_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos_theta = dot_product / norms_product
        return cos_theta

    def extract_features(self):
        features = []

        # 摄像机运动向量计算（需要poses和timestamps）
        camera_velocity = self.compute_velocity(np.array(self.poses), self.compute_timediff(self.timestamps))

        for i, label in enumerate(self.point_clouds):
            if label is None:
                continue  # 跳过无效的时间戳

            # 获取点云和位置数据
            point_cloud = label['point_cloud']
            position = np.array([label['position']['x'], label['position']['y'], label['position']['z']])
            pose = self.poses[i]

            # 计算边界框尺寸和体积
            obb_extent = self.compute_bounding_box(point_cloud)
            volume = np.prod(obb_extent)
            surface_area = 2 * (obb_extent[0]*obb_extent[1] + obb_extent[1]*obb_extent[2] + obb_extent[2]*obb_extent[0])
            xy_area = obb_extent[0] * obb_extent[1]

            # PCA 主轴
            pca_axes = self.compute_principal_axes(point_cloud)
            z_axis = np.array([0, 0, 1])
            principal_axis = pca_axes[:, 0]

            # 余弦相似度
            pca_z_cosine = self.compute_cosine_similarity(principal_axis, z_axis)

            # 其他特征
            principal_motion_cosine = self.compute_cosine_similarity(principal_axis, camera_velocity[i])
            camera_motion_cosine = self.compute_cosine_similarity(camera_velocity[i], pose['direction'])

            # 收集所有特征
            feature_vector = {
                'obb_extent': obb_extent,
                'volume': volume,
                'surface_area': surface_area,
                'xy_area': xy_area,
                'pca_z_cosine': pca_z_cosine,
                'principal_motion_cosine': principal_motion_cosine,
                'camera_motion_cosine': camera_motion_cosine,
                'camera_velocity': camera_velocity[i]
            }
            features.append(feature_vector)

        return features

    @staticmethod
    def compute_angle_change(velocity_vectors):
        angle_changes = []
        for i in range(1, len(velocity_vectors)):
            angle_change = ObjectFeaturesExtractor.compute_cosine_similarity(velocity_vectors[i-1], velocity_vectors[i])
            angle_changes.append(angle_change)
        return [angle_changes[0]] + angle_changes  # 第一个时间点没有前一个速度，可以标记为0或其他适当的值



class ObjectHandler:
    def __init__(self):
        pass

    @staticmethod
    def object_sequence(frame_data, target):
        sequence = []
        for frame in frame_data:
            is_found = False
            for label in frame['labels']:
                if label['object_id'] == target:
                    sequence.append(label)
                    is_found = True
                    break
            if not is_found:
                sequence.append(None)
        return sequence

    @staticmethod
    def timestamp_sequence(frame_data):
        return [frame['pose']['timestamp'] for frame in frame_data]

    @staticmethod
    def pose_sequence(frame_data):
        return [frame['pose']['position'] for frame in frame_data]

def test_data():
    base_dir = 'data'
    frame_id = '9048_3'

    processor = FrameDataProcessor(base_dir, frame_id)
    frame_data = processor.load_all_frame_data()
    poses = ObjectHandler.pose_sequence(frame_data)
    objs = ObjectHandler.object_sequence(frame_data, target=1)
    #[visualization([sequence[i]['point_cloud'], sequence[i]['bbox']]) if sequence[i] else None for i in range(len(sequence))]
    timestamps = ObjectHandler.timestamp_sequence(frame_data)
    feature = ObjectFeaturesExtractor(objs, poses, timestamps)
    feature.extract_features()

if __name__ == '__main__':
    test_data()
