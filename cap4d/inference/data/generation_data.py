import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cap4d.inference.data.inference_data import CAP4DInferenceDataset

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

def pivot_camera_intrinsic(extrinsics, target, angles, distance_factor=1.):
    """
    Rotates a camera around a target point.

    Parameters:
    - extrinsics: (4x4) numpy array, world_to_camera transformation matrix.
    - target: (3,) numpy array, target coordinates to pivot around.
    - angles: (3,) array-like, rotation angles (degrees) around X, Y, Z axes.

    Returns:
    - new_extrinsics: (4x4) numpy array, updated world_to_camera transformation.
    """
    extrinsics = np.linalg.inv(extrinsics)

    # Extract rotation and translation from extrinsics
    R_c2w = extrinsics[:3, :3]  # 3x3 rotation matrix
    t_c2w = extrinsics[:3, 3]   # 3x1 translation vector

    # Compute offset vector from target to camera
    v = (t_c2w - target) * distance_factor

    # Compute rotation matrix for given angles
    R_delta = R.from_euler('YX', angles, degrees=True).as_matrix()  # 'yx'

    # Apply intrinsic rotation to the camera's rotation (local frame)
    new_R_c2w = R_c2w @ R_delta

    # Rotate position offset in camera frame as well
    new_v = R_c2w @ R_delta @ np.linalg.inv(R_c2w) @ v
    new_t_c2w = target + new_v

    # Construct new extrinsics
    new_extrinsics = np.eye(4)
    new_extrinsics[:3, :3] = new_R_c2w
    new_extrinsics[:3, 3] = new_t_c2w

    return np.linalg.inv(new_extrinsics)


def elipsis_sample(yaw_limit, pitch_limit):
    if yaw_limit == 0. or pitch_limit == 0.:
        return 0., 0.
    
    dist = 1.
    while dist >= 1.:
        yaw = np.random.uniform(-yaw_limit, yaw_limit)
        pitch = np.random.uniform(-pitch_limit, pitch_limit)

        dist = np.sqrt((yaw / yaw_limit) ** 2 + (pitch / pitch_limit) ** 2)

    return yaw, pitch


def visualize_camera_extrinsics(flame_list, save_path=None, show_plot=True, title="Camera Extrinsics Visualization", target_point=None):
    """
    可视化flame_list中每一帧的相机外参(extr)
    
    参数:
    - flame_list: 包含相机外参的字典列表
    - save_path: 保存图片的路径（可选）
    - show_plot: 是否显示图片
    - title: 图片标题
    - target_point: 目标点位置（如ref_tra_cv），将标记在图中
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取所有相机的位置和朝向
    camera_positions = []
    camera_directions = []
    
    for i, flame_dict in enumerate(flame_list):
        extr = flame_dict["extr"][0]  # 取出4x4的外参矩阵
        
        # 外参矩阵是world_to_camera，我们需要camera_to_world来获取相机在世界坐标系中的位置
        cam_to_world = np.linalg.inv(extr)
        
        # 相机位置是变换矩阵的平移部分
        camera_pos = cam_to_world[:3, 3]
        camera_positions.append(camera_pos)
        
        # 相机朝向是-Z轴方向（OpenCV约定）
        camera_dir = -cam_to_world[:3, 2]  # -Z轴方向
        camera_directions.append(camera_dir)
    
    camera_positions = np.array(camera_positions)
    camera_directions = np.array(camera_directions)
    
    # 绘制相机位置（不显示颜色变化，统一颜色）
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
               c='blue', s=20, alpha=0.6, label='Cameras')
    
    # 绘制所有相机的朝向向量（精确显示每个相机的朝向）
    for i in range(len(camera_positions)):
        pos = camera_positions[i]
        dir_vec = camera_directions[i] * 0.3  # 稍微缩短向量长度避免过于密集
        ax.quiver(pos[0], pos[1], pos[2], 
                 dir_vec[0], dir_vec[1], dir_vec[2], 
                 color='red', alpha=0.5, arrow_length_ratio=0.1)
    
    # 添加原点作为参考
    ax.scatter([0], [0], [0], c='black', s=80, marker='o', label='World Origin')
    
    # 如果提供了target_point，则标记目标点
    if target_point is not None:
        ax.scatter([target_point[0]], [target_point[1]], [target_point[2]], 
                  c='green', s=120, marker='*', label='Target Point (ref_tra_cv)', edgecolors='darkgreen', linewidth=2)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\n({len(flame_list)} cameras)')
    
    # 动态设置坐标轴范围，确保所有点都清晰可见
    all_points = [camera_positions]
    
    # 添加原点
    all_points.append(np.array([[0, 0, 0]]))
    
    # 如果有目标点，也添加进去
    if target_point is not None:
        all_points.append(np.array([target_point]))
    
    # 合并所有点
    all_positions = np.vstack(all_points)
    
    # 计算包含所有点的边界，并添加一些边距
    margin_factor = 0.2  # 20%的边距
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
    
    max_range = max(x_range, y_range, z_range)
    margin = max_range * margin_factor
    
    ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
    ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
    ax.set_zlim(all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin)
    
    # 不再需要颜色条，因为不按帧索引着色
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化图片已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_camera_trajectory_2d(flame_list, save_path=None, show_plot=True):
    """
    绘制相机轨迹的2D俯视图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    camera_positions = []
    yaw_angles = []
    pitch_angles = []
    
    for flame_dict in enumerate(flame_list):
        extr = flame_dict[1]["extr"][0]
        cam_to_world = np.linalg.inv(extr)
        camera_pos = cam_to_world[:3, 3]
        camera_positions.append(camera_pos)
        
        # 从旋转矩阵提取欧拉角
        R_mat = cam_to_world[:3, :3]
        r = R.from_matrix(R_mat)
        euler_angles = r.as_euler('yxz', degrees=True)
        yaw_angles.append(euler_angles[0])
        pitch_angles.append(euler_angles[1])
    
    camera_positions = np.array(camera_positions)
    
    # 绘制XY平面轨迹（俯视图）
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], 
               c=range(len(camera_positions)), cmap='viridis', s=20)
    ax1.plot(camera_positions[:, 0], camera_positions[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax1.scatter([0], [0], c='red', s=100, marker='o', label='Origin')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Camera Trajectory (Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # 绘制角度变化
    frame_indices = range(len(yaw_angles))
    ax2.plot(frame_indices, yaw_angles, 'b-', label='Yaw', linewidth=1.5)
    ax2.plot(frame_indices, pitch_angles, 'r-', label='Pitch', linewidth=1.5)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Camera Rotation Angles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D轨迹图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


class GenerationDataset(CAP4DInferenceDataset):
    def __init__(
        self, 
        generation_data_path,
        reference_flame_item,
        n_samples=2,
        yaw_range=55,
        pitch_range=20,
        expr_factor=1.0,
        resolution=512,
        downsample_ratio=8,
    ):
        super().__init__(resolution, downsample_ratio)

        self.n_samples = n_samples
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.is_ref = False
        self.flame_dicts = self.init_flame_params(
            generation_data_path,
            reference_flame_item,
            n_samples,
            yaw_range,
            pitch_range,
            expr_factor,
        )

    def init_flame_params(
        self,
        generation_data_path,
        reference_flame_item,
        n_samples,
        yaw_range,
        pitch_range,
        expr_factor,
    ):
        gen_data = dict(np.load(generation_data_path))
        ref_extr = reference_flame_item["extr"] # if grandma dataset opengl camera input
        ref_shape = reference_flame_item["shape"]
        ref_fx = reference_flame_item["fx"]
        ref_fy = reference_flame_item["fy"]
        ref_cx = reference_flame_item["cx"]
        ref_cy = reference_flame_item["cy"]
        ref_resolution = reference_flame_item["resolutions"]
        ref_rot = reference_flame_item["rot"]# if grandma dataset opengl camera input
        ref_tra = reference_flame_item["tra"]# if grandma dataset opengl camera input
        ref_tra_cv = ref_tra.copy()
        ref_tra_cv[:, 1:] = -ref_tra_cv[:, 1:]  # p3d to opencv
        ref_tra_gl = ref_tra.copy()
        ref_tra_gl[..., 2] = -ref_tra_gl[..., 2]
        ref_target = ref_tra[0]
        flame_list = []

        assert n_samples <= len(gen_data["expr"]), "too many samples"
        for expr, eye_rot in zip(gen_data["expr"][:n_samples], gen_data["eye_rot"][:n_samples]):
            yaw, pitch = elipsis_sample(yaw_range, pitch_range)

            rotated_extr = pivot_camera_intrinsic(ref_extr[0], ref_target, [yaw, pitch])

            flame_dict = {
                "shape": ref_shape,
                "expr": expr[None] * expr_factor,
                "eye_rot": eye_rot[None] * expr_factor,
                "rot": ref_rot,
                "tra": ref_tra,
                "extr": rotated_extr[None],
                "resolutions": ref_resolution,
                "fx": ref_fx,
                "fy": ref_fy,
                "cx": ref_cx,
                "cy": ref_cy,
            }
            flame_list.append(flame_dict)

        self.flame_list = flame_list
        self.ref_extr = ref_extr[0]

        # visualize_camera_extrinsics(flame_list, save_path="debug/gen_camera_extrinsics.png", show_plot=True, target_point=ref_target)
        # plot_camera_trajectory_2d(flame_list, save_path="debug/gen_camera_trajectory.png", show_plot=True)
        
    def visualize_cameras(self, save_path=None, show_plot=True):
        """
        可视化当前数据集中所有相机的外参
        """
        return visualize_camera_extrinsics(self.flame_list, save_path, show_plot, 
                                         f"Generation Dataset Camera Extrinsics ({len(self.flame_list)} frames)")
    
    def plot_trajectory(self, save_path=None, show_plot=True):
        """
        绘制相机轨迹的2D图
        """
        return plot_camera_trajectory_2d(self.flame_list, save_path, show_plot)
