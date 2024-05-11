import os
import numpy as np
import open3d as o3d


class PointCloudLoader:
    def __init__(self, base_dir):
        """
        初始化PointCloudLoader
        :param base_dir: 存储点云文件的基本目录
        """
        self.base_dir = base_dir

    def construct_file_path(self, track_id, frame_id):
        """
        构造PCD文件的完整路径
        :param track_id: 跟踪数据集的ID（1, 2, 或 3）
        :param frame_id: 帧的编号，如9048_1_frame
        :return: 完整的文件路径
        """
        file_path = f"{self.base_dir}/tracking_train_pcd_{track_id}/result_{frame_id}_frame"
        return [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pcd')]

    @staticmethod
    def load_point_cloud(file_path):
        """
        读取单个PCD文件
        :param file_path: PCD文件的完整路径
        :return: 加载的点云对象
        """
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def load_all_point_clouds(self, track_id, frame_id):
        """
        加载一个跟踪帧中的所有PCD文件
        :param track_id: 跟踪数据集的ID
        :param frame_id: 帧的编号
        :return: 点云对象列表
        """
        file_paths = self.construct_file_path(track_id, frame_id)
        point_clouds = [self.load_point_cloud(fp) for fp in file_paths]
        return point_clouds

    @staticmethod
    def convert_to_numpy(point_clouds):
        return [np.asarray(point_clouds[i].points) for i in range(len(point_clouds))]


class CameraPoseLoader:
    def __init__(self, base_dir):
        """
        初始化CameraPoseLoader
        :param base_dir: 存储摄像头位置信息文件的基本目录
        """
        self.base_dir = base_dir

    def construct_file_path(self, frame_id, file_id):
        """
        构造摄像头位置信息文件的完整路径
        :param frame_id: 帧的编号，如9048_1_frame
        :param file_id: 文件编号，如233
        :return: 完整的文件路径
        """
        file_path = f"{self.base_dir}/tracking_train_pose/result_{frame_id}_frame/{file_id}_pose.txt"
        return file_path

    @staticmethod
    def read_pose_file(file_path):
        """
        读取单个摄像头位置信息文件
        :param file_path: 摄像头位置信息文件的完整路径
        :return: 摄像头位置和姿态数据的字典
        """
        with open(file_path, 'r') as file:
            data = file.readline().strip().split()
            pose_data = {
                'timestamp': float(data[1]),
                'position': {'x': float(data[2]), 'y': float(data[3]), 'z': float(data[4])},
                'quaternion': {'x': float(data[5]), 'y': float(data[6]), 'z': float(data[7]), 'w': float(data[8])}
            }
        return pose_data

    def load_camera_poses(self, frame_id, file_id):
        """
        加载指定的摄像头位置信息
        :param frame_id: 帧的编号
        :param file_id: 文件编号
        :return: 摄像头位置和姿态数据
        """
        file_path = self.construct_file_path(frame_id, file_id)
        return self.read_pose_file(file_path)


class ObjectLabelLoader:
    def __init__(self, base_dir):
        """
        初始化ObjectLabelLoader
        :param base_dir: 存储标签信息文件的基本目录
        """
        self.base_dir = base_dir

    def construct_file_path(self, frame_id, file_id):
        """
        构造标签信息文件的完整路径
        :param frame_id: 帧的编号，如9048_1
        :param file_id: 文件编号，如233
        :return: 完整的文件路径
        """
        file_path = f"{self.base_dir}/tracking_train_label/{frame_id}/{file_id}.txt"
        return file_path

    @staticmethod
    def read_label_file(file_path):
        """
        读取单个标签信息文件
        :param file_path: 标签信息文件的完整路径
        :return: 包含所有物体标签数据的列表
        """
        objects = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                label_data = {
                    'object_id': int(data[0]),
                    'object_type': int(data[1]),
                    'position': {'x': float(data[2]), 'y': float(data[3]), 'z': float(data[4])},
                    'size': {'length': float(data[5]), 'width': float(data[6]), 'height': float(data[7])},
                    'heading': float(data[8])
                }
                objects.append(label_data)
        return objects

    def load_object_labels(self, frame_id, file_id):
        """
        加载指定的标签信息
        :param frame_id: 帧的编号
        :param file_id: 文件编号
        :return: 包含所有物体标签的列表
        """
        file_path = self.construct_file_path(frame_id, file_id)
        return self.read_label_file(file_path)

    @staticmethod
    def classification_objects(objs, target):
        """
        加载指定的类别
        :param objs: 对象字典
        :param target: 目标类别' 1 for small vehicles, 2 for big vehicles, 3 for pedestrian, 4 for motorcyclist and bicyclist, 5 for traffic cones and 6 for others. '
        :return: 包含目标对象的列表
        """
        return [obj for obj in objs if objs['object_type'] == target]


class FrameDataProcessor:
    def __init__(self, base_dir, frame_id):
        """
        初始化FrameDataProcessor
        :param base_dir: 数据的根目录
        :param frame_id: 当前处理的帧ID，例如 '9048_1_frame'
        """
        self.base_dir = base_dir
        self.frame_id = frame_id
        self.pcd_loader = PointCloudLoader(base_dir)
        self.pose_loader = CameraPoseLoader(base_dir)
        self.label_loader = ObjectLabelLoader(base_dir)

    def load_frame_data(self, file_id):
        """
        加载一个帧的所有相关数据
        :param file_id: 文件编号，如 '233'
        :return: 返回一个包含pose, labels和point_clouds的字典
        """
        # 加载Pose数据
        pose = self.pose_loader.load_camera_poses(self.frame_id, file_id)

        # 加载Label数据
        labels = self.label_loader.load_object_labels(self.frame_id, file_id)

        # 确定PCD文件所在的track_id文件夹
        track_id = self.find_track_id()
        if track_id is None:
            raise ValueError("Frame ID not found in any track ID folder")

        # 加载与file_id对应的单个PCD文件
        pcd_path = os.path.join(self.base_dir, f'tracking_train_pcd_{track_id}',
                                f'result_{self.frame_id}_frame', f'{file_id}.pcd')

        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"No PCD file found for file ID {file_id} in frame {self.frame_id}")
        point_cloud = self.pcd_loader.load_point_cloud(pcd_path)

        return {
            'pose': pose,
            'labels': labels,
            'point_cloud': point_cloud  # 返回单个点云对象
        }

    def find_track_id(self):
        """
        确定frame_id位于哪个track_id下
        """
        track_ids = [1, 2, 3]
        for track_id in track_ids:
            pcd_path = os.path.join(self.base_dir, f'tracking_train_pcd_{track_id}', f'result_{self.frame_id}_frame')
            if os.path.exists(pcd_path):
                return track_id
        return None

    def load_all_frame_data(self):
        label_path = os.path.join(self.base_dir, 'tracking_train_label', self.frame_id)
        file_ids = [f.split('_')[0][:-4] for f in os.listdir(label_path) if f.endswith('.txt')]
        all_data = []

        for file_id in file_ids:
            frame_data = self.load_frame_data(file_id)
            self.collect_object_points(frame_data)
            frame_data = self.traverse_points(frame_data, file_id)
            all_data.append(frame_data)
        temp = []
        for frame_data in all_data:
            temp += [frame_data['point_cloud']] + [label['bbox'] for label in frame_data['labels']]
        visualization(temp)

        return all_data

    @staticmethod
    def collect_object_points(frame_data):
        pc_manager = PointCloudManager(frame_data['point_cloud'])
        # 提取对象的点云（在坐标转换之前）
        pc_manager.extract_object_point_clouds(frame_data['labels'])

    def traverse_points(self, frame_data, file_id):
        try:
            # Convert the entire point cloud to global coordinates
            point_cloud_abs = self.transform_to_global_coordinates(frame_data['pose'],
                                                                   np.asarray(frame_data['point_cloud'].points))
            frame_data['point_cloud'].points = o3d.utility.Vector3dVector(point_cloud_abs)

            # Process each label for object point cloud, coordinates, heading, and bounding box
            for label in frame_data['labels']:
                # Convert object point cloud to global coordinates
                object_point_cloud = label['point_cloud']
                global_object_points = self.transform_to_global_coordinates(frame_data['pose'],
                                                                            np.asarray(object_point_cloud.points))
                object_point_cloud.points = o3d.utility.Vector3dVector(global_object_points)

                # Convert label position to global coordinates
                pos = np.array([[label['position']['x'], label['position']['y'], label['position']['z']]])
                transformed_pos = self.transform_to_global_coordinates(frame_data['pose'], pos)
                label['position']['x'], label['position']['y'], label['position']['z'] = transformed_pos[0]

                # Rotate heading
                label['heading'] = self.rotate_heading(frame_data['pose']['quaternion'], label['heading'])

                # Transform bbox object to global coordinates (if it exists)
                if 'bbox' in label:
                    bbox = label['bbox']
                    quaternion_array = np.array([frame_data['pose']['quaternion']['w'],
                                                 frame_data['pose']['quaternion']['x'],
                                                 frame_data['pose']['quaternion']['y'],
                                                 frame_data['pose']['quaternion']['z']]).reshape(4, 1)
                    global_rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion_array)
                    bbox_center_global = self.transform_to_global_coordinates(frame_data['pose'],
                                                                              np.array([bbox.center]))
                    bbox.center = bbox_center_global[0]
                    bbox.R = global_rotation_matrix @ bbox.R

            return frame_data


        except FileNotFoundError as e:
            print(f"Error loading data for file ID {file_id}: {e}")

    @staticmethod
    def transform_to_global_coordinates(pose, points):
        """
        将点从相对坐标转换到全局坐标系
        :param pose: 包含位置和四元数的字典
        :param points: 待转换的点数组
        :return: 转换后的点数组
        """
        # 提取位置和四元数
        translation = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
        quaternion = np.array(
            [pose['quaternion']['w'], pose['quaternion']['x'], pose['quaternion']['y'], pose['quaternion']['z']])

        # 创建旋转矩阵
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

        # 应用旋转和平移
        points_transformed = np.dot(points, rotation_matrix.T) + translation
        return points_transformed

    @staticmethod
    def rotate_heading(quaternion, heading):
        """
        使用四元数旋转2D heading角度
        :param quaternion: 摄像机姿势信息
        :param heading: 本地坐标系下的heading（角度值，假设0度指向正北，角度逆时针增加）
        :return: 全局坐标系下的heading（角度值）
        """
        # 将2D heading角度转换为向量
        quaternion = np.array(
            [quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']])

        heading_vector = np.array([np.cos(heading), np.sin(heading), 0])

        # 创建旋转矩阵
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

        # 旋转heading向量
        global_heading_vector = np.dot(rotation_matrix, heading_vector)

        # 计算全局heading角度
        global_heading = np.arctan2(global_heading_vector[1], global_heading_vector[0])
        return global_heading


class PointCloudManager:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud

    @staticmethod
    def create_3d_bbox(position, size, heading):
        """
        Create an OrientedBoundingBox for an object.
        :param position: The global position of the object's center (x, y, z).
        :param size: The dimensions of the object as a dictionary (length, width, height).
        :param heading: The heading angle of the object around the z-axis in radians.
        :return: An OrientedBoundingBox object.
        """
        center = np.array([position['x'], position['y'], position['z']])

        # If size is a dictionary, convert it to an array and divide by 2
        if isinstance(size, dict):
            extent = np.array([size['length'], size['width'], size['height']])
        else:
            extent = np.array(size) / 2  # Assuming size is already in an array/list form

        # Compute the rotation matrix from the heading angle
        R = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        # Create an oriented bounding box with the correct orientation and extents
        bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
        return bbox

    def filter_points_in_bbox(self, bbox):
        """
        Filter points that are within the bounding box.
        :param bbox: An OrientedBoundingBox object.
        :return: A point cloud subset that lies within the bbox.
        """
        # Use the bounding box to crop the point cloud
        cropped_point_cloud = self.point_cloud.crop(bbox)
        return cropped_point_cloud

    def extract_object_point_clouds(self, labels):
        """
        Extract point clouds for each object based on its bounding box.
        :param labels: A list of label dictionaries containing position, size, and heading.
        """
        bboxes = []
        for label in labels:
            bbox = self.create_3d_bbox(label['position'], label['size'], label['heading'])
            bboxes.append(bbox)

            object_points = self.filter_points_in_bbox(bbox)
            # Ensure to convert the cropped point cloud back to Open3D PointCloud if necessary
            label['point_cloud'] = object_points if isinstance(object_points,
                                                               o3d.geometry.PointCloud) else o3d.geometry.PointCloud(
                object_points)
            label['bbox'] = bbox
        # visualization([self.point_cloud] + bboxes)
        return


def visualization(objs):
    o3d.visualization.draw_geometries(objs)
    return


def test_PCD():
    loader = PointCloudLoader(base_dir='data')
    point_clouds = loader.load_all_point_clouds(track_id=1, frame_id='9048_1')
    point_cloud_np = loader.convert_to_numpy(point_clouds)
    print(point_cloud_np)


def test_pose():
    pose_loader = CameraPoseLoader(base_dir='data')
    camera_pose = pose_loader.load_camera_poses(frame_id='9048_1', file_id='233')
    print(camera_pose)


def test_label():
    label_loader = ObjectLabelLoader(base_dir='data')
    object_labels = label_loader.load_object_labels(frame_id='9048_1', file_id='233')
    print(object_labels)


def test_frame():
    base_dir = 'data'
    frame_id = '9048_3'

    processor = FrameDataProcessor(base_dir, frame_id)
    frame_data = processor.load_all_frame_data()

    for i, frame in enumerate(frame_data):
        print(f'for No.{i} frame we have:')
        print('Pose of this PCD files is', frame['pose'])
        print('Labels found in this PCD files are', len(frame['labels']))
        print('First label coordinates are', frame['labels'])
        print('PCD file is composed of', frame['point_cloud'])  # 注意现在返回的是单个点云对象
        print('First point coordinates are', np.asarray(frame['point_cloud'].points)[0])
        print('+++++++++++++++++end++++++++++++++++')  # 注意现在返回的是单个点云对象


# 测试
if __name__ == '__main__':
    # test_PCD()
    # test_pose()
    # test_label()
    test_frame()
