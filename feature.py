import numpy as np
from data import test_extract

class ObjectHandler:
    def __init__(self, frame_data):
        self.frame_data = frame_data
        self.poses = self.pose_sequence(frame_data)
        self.times = self.timestamp_sequence(frame_data)
        (self.camera_movement, self.camera_velocity, self.camera_acceleration,
         self.camera_angle_change, self.camera_angle_change_speed) = self.camera_movement_sequence()
        self.obj = []

    def object_sequence(self, target):
        sequence = []
        for frame in self.frame_data:
            is_found = False
            for label in frame['labels']:
                if label['object_id'] == target:
                    sequence.append(label)
                    is_found = True
                    break
            if not is_found:
                sequence.append(None)
        self.obj = sequence

    @staticmethod
    def timestamp_sequence(frame_data):
        return [frame['pose']['timestamp'] for frame in frame_data]

    @staticmethod
    def pose_sequence(frame_data):
        return [frame['pose']['position'] for frame in frame_data]

    def camera_movement_sequence(self):
        camera_location_vectors = np.array([[point['x'], point['y'], point['z']] for point in self.poses])
        camera_time_vector = self.times
        return ObjectFeaturesExtractor.compute_movement(camera_location_vectors, camera_time_vector)

    @staticmethod
    def all_obj(frame_data):
        unique_object_ids = set()

        # 遍历列表中的每个元素（每个元素都是一个字典）
        for item in frame_data:
            # 在每个字典中访问'labels'键，它的值是一个列表
            labels = item['labels']
            # 遍历'labels'列表中的每个元素（每个元素都是一个字典）
            for label in labels:
                # 在每个字典中访问'object_id'键，并将其添加到集合中
                object_id = label['object_id']
                unique_object_ids.add(object_id)

        # 打印所有不重复的object_id
        return unique_object_ids


class ObjectFeaturesExtractor:
    def __init__(self, handler: ObjectHandler):
        """
        初始化特征提取器
        :param handler: 对象处理器
        """
        self.handler = handler
        self.point_clouds = handler.obj
        self.poses = handler.poses
        self.timestamps = handler.times
        self.camera_movement, self.camera_velocity, self.camera_acceleration = (
            handler.camera_movement, handler.camera_velocity, handler.camera_acceleration)

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

    @staticmethod
    def compute_density(point_cloud, volume):
        """
        计算点云的密度
        """
        points = np.asarray(point_cloud.points).shape[0]
        return points / volume if volume != 0 else 0

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
    def compute_velocity(position_diff_vectors, extended_time_intervals):
        # 由于 extended_time_intervals 已经与原始 timestamps 等长，我们直接使用整个数组进行广播
        return position_diff_vectors / extended_time_intervals[:, np.newaxis]

    @staticmethod
    def compute_acceleration(velocity_diff_vectors, extended_time_intervals):
        return velocity_diff_vectors / extended_time_intervals[:, np.newaxis]

    @staticmethod
    def compute_cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norms_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos_theta = dot_product / norms_product
        return cos_theta

    @staticmethod
    def compute_movement(location_vectors, time_vectors):
        self = ObjectFeaturesExtractor
        movement = self.compute_diff(location_vectors)
        time_diff = self.compute_timediff(time_vectors)

        velocity = self.compute_velocity(movement, time_diff)
        velocity_diff = self.compute_diff(velocity)
        acceleration = self.compute_acceleration(velocity_diff, time_diff)

        angel_change = self.compute_angle_change(velocity)
        angel_change_speed = angel_change / time_diff

        return movement, velocity, acceleration, angel_change, angel_change_speed

    def extract_features(self):
        features = []

        # 摄像机运动向量计算（需要poses和timestamps）

        label_sequence = []
        for i, label in enumerate(self.point_clouds):
            if label is None:
                continue  # 跳过无效的时间戳
            label_sequence.append([i, label])

        (acceleration_magnitudes, obj_accelerations, obj_angle_changes,
         obj_angle_speeds, obj_movements, obj_velocities, velocity_magnitudes) = self.obj_movement_sequence(
            label_sequence)

        for label_id, (i, label) in enumerate(label_sequence):
            # 获取点云和位置数据
            point_cloud = label['point_cloud']

            density, obb_extent, surface_area, volume = self.compute_size(point_cloud)

            density /= self.distance_between_points([self.poses[i], label['position']]) ** 2

            pca_z_cosine, principal_axis = self.compute_main_axis(point_cloud)

            height, length, width, xy_area = self.determine_3d_value(obb_extent, pca_z_cosine)

            (acceleration_magnitude, angle_change, angle_speed, obj_acceleration, obj_velocity,
             principal_camera_cosine, velocity_camera_cosine, velocity_magnitude) = self.computer_obj_motion(
                acceleration_magnitudes, i, label_id, obj_accelerations, obj_angle_changes, obj_angle_speeds,
                obj_movements, obj_velocities, principal_axis, velocity_magnitudes)

            # 收集所有特征
            feature_vector = {
                'length': length,
                'width': width,
                'height': height,
                'volume': volume,
                'surface_area': surface_area,
                'xy_area': xy_area,
                'density': density,
                'pca_z_cosine': pca_z_cosine,
                'principal_camera_cosine': principal_camera_cosine,
                'velocity_camera_cosine': velocity_camera_cosine,
                'velocity_magnitude': velocity_magnitude,
                'acceleration_magnitude': acceleration_magnitude,
                'object_type': label['object_type'],
                'angle_change': angle_change,
                'angle_speed': angle_speed
            }
            features.append(feature_vector)

        return features

    def computer_obj_motion(self, acceleration_magnitudes, i, label_id, obj_accelerations, obj_angle_changes,
                            obj_angle_speeds, obj_movements, obj_velocities, principal_axis, velocity_magnitudes):
        # 速度与加速度的向量
        obj_movement, obj_velocity, obj_acceleration = (
            obj_movements[label_id], obj_velocities[label_id], obj_accelerations[label_id])
        # 速度,加速度,角变化,角速度的值
        (velocity_magnitude, acceleration_magnitude,
         angle_change, angle_speed) = (velocity_magnitudes[label_id], acceleration_magnitudes[label_id],
                                       obj_angle_changes[label_id], obj_angle_speeds[label_id])
        # 速度的方向与自身、相机的方向余弦相似度
        principal_camera_cosine = abs(self.compute_cosine_similarity(principal_axis, self.camera_velocity[i]))
        velocity_camera_cosine = abs(self.compute_cosine_similarity(obj_velocity, self.camera_velocity[i]))
        return (acceleration_magnitude, angle_change, angle_speed, obj_acceleration,
                obj_velocity, principal_camera_cosine, velocity_camera_cosine, velocity_magnitude)

    def obj_movement_sequence(self, label_sequence):
        object_location = np.array([[label['position']['x'], label['position']['y'], label['position']['z']]
                                    for _, label in label_sequence])
        object_time = [self.timestamps[i] for i, _ in label_sequence]
        obj_movements, obj_velocities, obj_accelerations, obj_angle_changes, obj_angle_speeds = (
            self.compute_movement(object_location, object_time))
        velocity_magnitudes = np.linalg.norm(obj_velocities, axis=1)
        acceleration_magnitudes = np.linalg.norm(obj_accelerations, axis=1)
        return (acceleration_magnitudes, obj_accelerations, obj_angle_changes,
                obj_angle_speeds, obj_movements, obj_velocities, velocity_magnitudes)

    @staticmethod
    def determine_3d_value(obb_extent, pca_z_cosine):
        if pca_z_cosine >= 0.5:
            length, width, height = obb_extent[1], obb_extent[2], obb_extent[0]
        else:
            length, width, height = obb_extent[0], obb_extent[1], obb_extent[2]
        xy_area = length * width
        return height, length, width, xy_area

    @staticmethod
    def distance_between_points(points):
        """
        计算两点之间的距离
        参数:
        points：两点list
        返回:
        float, 两点之间的距离
        """
        points = [[point["x"], point["y"], point["z"]] if type(point) is dict else point for point in points]

        # 将点的坐标转换为numpy数组
        point1 = np.array(points[0])
        point2 = np.array(points[1])

        # 计算两点之间的欧氏距离
        distance = np.linalg.norm(point1 - point2)

        return distance

    @staticmethod
    def compute_main_axis(point_cloud):
        # PCA 主轴
        self = ObjectFeaturesExtractor
        pca_axes = self.compute_principal_axes(point_cloud)
        z_axis = np.array([0, 0, 1])
        principal_axis = pca_axes[:, 0]
        # 主轴与Z轴的余弦相似度
        pca_z_cosine = abs(self.compute_cosine_similarity(principal_axis, z_axis))
        return pca_z_cosine, principal_axis

    @staticmethod
    def compute_size(point_cloud):
        # 计算边界框尺寸和体积
        self = ObjectFeaturesExtractor
        obb_extent = self.compute_bounding_box(point_cloud)
        volume = np.prod(obb_extent)
        surface_area = 2 * (
                obb_extent[0] * obb_extent[1] + obb_extent[1] * obb_extent[2] + obb_extent[2] * obb_extent[0])
        density = self.compute_density(point_cloud, volume)
        return density, obb_extent, surface_area, volume

    @staticmethod
    def compute_angle_change(velocity_vectors):
        angle_changes = []
        for i in range(1, len(velocity_vectors)):
            angle_change = ObjectFeaturesExtractor.compute_cosine_similarity(velocity_vectors[i - 1],
                                                                             velocity_vectors[i])
            angle_changes.append(angle_change)
        return np.array([angle_changes[0]] + angle_changes)  # 第一个时间点没有前一个速度，可以标记为0或其他适当的值


class StaticFeaturesExtractor:
    def __init__(self, dataset):
        """
        初始化静态特征提取器
        :param dataset: 结构化数据
        """
        self.tools = ObjectFeaturesExtractor
        self.frame_data = dataset

    def extract_features(self):
        """
        提取所有点云的静态特征
        :return: 包含每个对象特征的列表
        """

        features_list = []
        for data in self.frame_data:
            features = self.compute_static_features(data)
            features_list.append(features)
        return features_list

    @staticmethod
    def compute_geometric_center(point_cloud):
        """
        计算点云的几何中心。
        :param point_cloud: Open3D 点云对象。
        :return: 几何中心的坐标 (numpy.ndarray)。
        """
        # 将点云的点转换为 NumPy 数组
        points = np.asarray(point_cloud.points)

        # 计算所有点的平均位置
        geometric_center = np.mean(points, axis=0)

        return geometric_center

    def compute_static_features(self, dataset):
        """
        利用已有方法计算单个点云的静态特征
        :param dataset: 单个对象的点云
        :return: 特征字典
        """
        features_list = []
        for obj in dataset['extract_obj']:
            if np.asarray(obj.points).shape[0] < 5:
                continue

            features = {}
            # 调用 ObjectFeaturesExtractor 的方法计算边界框尺寸

            pca_z_cosine, principal_axis = self.tools.compute_main_axis(obj)

            density, obb_extent, surface_area, volume = self.tools.compute_size(obj)

            center_location = self.compute_geometric_center(obj)
            camera_location = dataset['pose']['position']

            density /= self.tools.distance_between_points([camera_location, center_location]) ** 2

            height, length, width, xy_area = self.tools.determine_3d_value(obb_extent, pca_z_cosine)

            # 宽度，高度，长度
            features['width'], features['height'], features['length'] = width, height, length
            # 体积
            features['volume'] = volume
            # 表面积
            features['surface_area'] = surface_area
            # XY平面面积
            features['xy_area'] = xy_area
            # 密度
            features['density'] = density
            # 计算PCA的Z轴余弦相似度
            features['pca_z_cosine'] = pca_z_cosine

            features['location_x'] = center_location[0]
            features['location_y'] = center_location[1]
            features['location_z'] = center_location[2]

            features['location_x'] = center_location[0]
            features['location_y'] = center_location[1]
            features['location_z'] = center_location[2]
            features['timestamp'] = center_location[2]
            features_list.append(dataset['pose']['timestamp'])

        return features_list


if __name__ == '__main__':
    frame_data = test_extract()
    static_extractor = StaticFeaturesExtractor(frame_data)
    static_extractor.extract_features()
