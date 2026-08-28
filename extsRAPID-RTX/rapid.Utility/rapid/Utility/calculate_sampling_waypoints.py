import math
import numpy as np
from typing import List


def calculate_fov_from_camera(focal_length, h_aperture, pixels):
    """
    航点计算辅助函数
    根据焦距、传感器水平孔径和像素比例计算水平和垂直视场角 (FOV)
    返回: (fov_h_deg, fov_v_deg)
    """
    # 1. 计算像素比例 (Aspect Ratio)
    # 2. 计算垂直孔径 (Vertical Aperture)
    # 修正：垂直孔径 = 水平孔径 * (高像素 / 宽像素)
    v_aperture = h_aperture * (pixels[1] / pixels[0])

    # 3. 计算视场角 (弧度)
    fov_h = 2 * math.atan(h_aperture / (2 * focal_length))
    fov_v = 2 * math.atan(v_aperture / (2 * focal_length))

    return math.degrees(fov_h), math.degrees(fov_v)


def calculate_footprint_from_camera(focal_length, h_aperture, pixels, altitude):
    """
    航点计算辅助函数
    计算相机在特定高度下的地面覆盖范围 (Footprint)
    参数:
        altitude: 飞行高度
    返回:
        (ground_width, ground_height) - 地面覆盖的宽和高
    """
    # 获取视场角
    fov_h_deg, fov_v_deg = calculate_fov_from_camera(focal_length, h_aperture, pixels)

    # 根据三角函数计算地面实际尺寸
    # Ground Dimension = 2 * altitude * tan(FOV / 2)
    # 也可以直接利用相似三角形原理: Ground_W = altitude * (h_aperture / focal_length)
    ground_w = 2 * altitude * math.tan(math.radians(fov_h_deg / 2))
    ground_h = 2 * altitude * math.tan(math.radians(fov_v_deg / 2))

    return (ground_w, ground_h)


def calculate_constant_altitude_sampling_waypoints(
        start_point=(0.0, 0.0),
        end_point=(100.0, 100.0),
        altitude=100,
        pixels=(1920, 1080),
        focal_length=50,
        horizontal_aperture=36,
        overlap_f_s=[80, 60],
        sensor_type='Perspective'):
    """
    计算航点列表, 光学第二种恒定高度飞行模拟, 产生在同一水平上的若干航点
    overlap_f_s: [前向重叠率%, 侧向重叠率%] (0-100)
    """
    print('1111111111111111111111111111111111111111111111')
    print(f'start_point{start_point},end_point{end_point},altitude{altitude},pixels{pixels},focal_length{focal_length}.horizontal_aperture{horizontal_aperture},overlap_f_s{overlap_f_s},sensor_type{sensor_type}')
    print('1111111111111111111111111111111111111111111111')
    # 1. 调用独立函数计算地面覆盖范围
    if sensor_type == 'Perspective':
        ground_w, ground_h = calculate_footprint_from_camera(
            focal_length, horizontal_aperture, pixels, altitude
        )
    elif sensor_type == 'Orthographic':
        ground_h = horizontal_aperture/10
        ground_w = ground_h * (pixels[1] / pixels[0])

    # 2. 计算步进距离 (Step Distance)
    # 航向步进 (Forward Step): 沿高度方向 H
    dist_step_f = ground_h * (1.0 - overlap_f_s[0] * 0.01)
    # 旁向步进 (Side Step): 沿宽度方向 W
    dist_step_s = ground_w * (1.0 - overlap_f_s[1] * 0.01)

    # 3. 确定区域边界
    min_x, max_x = min(start_point[0], end_point[0]), max(start_point[0], end_point[0])
    min_y, max_y = min(start_point[1], end_point[1]), max(start_point[1], end_point[1])

    # 4. 生成网格坐标 (从起点开始，确保覆盖到终点)
    x_coords = np.arange(min_x, max_x + dist_step_s, dist_step_s)
    y_coords = np.arange(min_y, max_y + dist_step_f, dist_step_f)

    # 存储传感器位置的目标位置
    waypoints = []
    target_position = []

    # 5. 采用弓字型 (S-Curve) 路径
    for i, x in enumerate(x_coords):
        # 偶数列从小到大，奇数列从大到小
        current_y_strip = y_coords if i % 2 == 0 else y_coords[::-1]
        for y in current_y_strip:
            waypoints.append((x, y, altitude))
            target_position.append((x, y, 0))

    return waypoints, (ground_w, ground_h), target_position


def calculate_semi_circular_sampling_waypoints(
    height=40.0,
    azimuth_deg=0.0,
    angle_range=[0.0, 90.0],
    step=1.0,
    center=[0.0, 0.0, 0.0],
    uav_offset=0.2,
):
    """
    在竖直半圆弧上采样位置
    :param height: 采样半径
    :param azimuth_deg: 方位角 (度)
    :param angle_range: 天顶角范围 [start, end]，0为正上方
    :param step: 步长绝对值 (度)
    :param center: 采样圆弧的圆心坐标 [x, y, z]
    :param uav_offset: 无人机相对于相机的Z轴偏移
    """

    start_angle = angle_range[0]
    end_angle = angle_range[1]

    # 自动判断步长的正负方向
    # 如果 start > end，步长应为负；反之为正
    actual_step = -abs(step) if start_angle > end_angle else abs(step)

    # 生成角度序列
    # 使用 np.arange，注意它不包含终点。如果需要包含终点，可以将 end_angle 加上一个极小值
    zenith_angles_deg = np.arange(start_angle, end_angle, actual_step)

    # 如果 arange 因为逻辑问题依然为空（例如 start == end），返回空列表
    if len(zenith_angles_deg) == 0:
        print("Warning: Generated waypoint list is empty. Check angle_range and step.")
        return [], []

    zenith_angles = np.deg2rad(zenith_angles_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # 1. 根据球坐标系公式计算 (相对坐标)
    # r_xy 是点在 XY 平面上的投影长度
    r_xy = height * np.sin(zenith_angles)
    x_rel = r_xy * np.cos(azimuth_rad)
    y_rel = r_xy * np.sin(azimuth_rad)
    z_rel = height * np.cos(zenith_angles)

    # 2. 构造绝对坐标 (加上 center)
    center = np.array(center)
    # 合并为 (N, 3) 矩阵并加偏移
    rel_coords = np.stack((x_rel, y_rel, z_rel), axis=-1)
    camera_pos_abs = rel_coords + center

    # 3. 计算无人机位置
    uav_pos_abs = camera_pos_abs.copy()
    uav_pos_abs[:, 2] += uav_offset  # 在绝对Z基础上增加偏移

    # --- 转换为元组列表 ---
    camera_pos_list = [tuple(v for v in point) for point in camera_pos_abs]
    uav_pos_list = [tuple(v for v in point) for point in uav_pos_abs]

    # 创建一个等长的观测目标位置的列表
    target_point_tuple = tuple(center)
    target_pos_list = [target_point_tuple] * len(camera_pos_list)

    return camera_pos_list, uav_pos_list, target_pos_list


def calculate_omnidirectional_sampling_waypoints(
    height=40.0,
    zenith_deg_list=[45.0],  # 现在接收一个列表或 FloatArray
    step=5.0,
    center=[0.0, 0.0, 0.0],
    azimuth_range=[0.0, 360.0],
    uav_offset=0.2
):
    """
    全方位（多层水平环）采样位置计算
    :param height: 采样半径 (r)
    :param zenith_deg_list: 天顶角列表 [z1, z2, ...], 0为正上方
    :param step: 方位角步长 (度)
    :param center: 采样圆心坐标 [x, y, z]
    :param azimuth_range: 方位角范围 [start, end]
    :param uav_offset: 无人机相对于相机的Z轴偏移
    """

    # 1. 准备方位角序列 (Azimuth)
    start_azi = azimuth_range[0]
    end_azi = azimuth_range[1]
    actual_step = -abs(step) if start_azi > end_azi else abs(step)

    # 包含终点处理
    azimuth_angles_deg = np.arange(start_azi, end_azi + (actual_step * 0.1), actual_step)

    if len(azimuth_angles_deg) == 0 or len(zenith_deg_list) == 0:
        return [], []

    # 转换为弧度
    azimuth_rad = np.deg2rad(azimuth_angles_deg)      # 形状: (N,)
    zenith_rad = np.deg2rad(np.array(zenith_deg_list)) # 形状: (M,)

    # 2. 利用广播机制计算所有组合
    # 计算 Z 轴高度：对于每个天顶角有一个固定的 Z
    # z_rel 形状: (M,)
    z_rel = height * np.cos(zenith_rad)

    # 计算点在 XY 平面上的投影半径
    # r_xy 形状: (M,)
    r_xy = height * np.sin(zenith_rad)

    # 计算 X 和 Y
    # 利用 np.outer 或 维度重塑进行广播: (M, 1) * (1, N) -> (M, N)
    x_rel = r_xy[:, np.newaxis] * np.cos(azimuth_rad)[np.newaxis, :]
    y_rel = r_xy[:, np.newaxis] * np.sin(azimuth_rad)[np.newaxis, :]

    # 将 z_rel 扩展到与 x, y 相同的形状 (M, N)
    z_rel_expanded = np.repeat(z_rel[:, np.newaxis], len(azimuth_rad), axis=1)

    # 3. 展平并构造绝对坐标
    # 将 (M, N) 的矩阵拉直为长度为 M*N 的一维数组
    x_flat = x_rel.flatten()
    y_flat = y_rel.flatten()
    z_flat = z_rel_expanded.flatten()

    # 构造 (M*N, 3) 矩阵
    center = np.array(center)
    rel_coords = np.stack((x_flat, y_flat, z_flat), axis=-1)
    camera_pos_abs = rel_coords + center

    # 4. 计算无人机位置
    uav_pos_abs = camera_pos_abs.copy()
    uav_pos_abs[:, 2] += uav_offset

    # 5. 转换为元组列表并确保类型为原生 float
    camera_pos_list = [tuple(float(v) for v in point) for point in camera_pos_abs]
    uav_pos_list = [tuple(float(v) for v in point) for point in uav_pos_abs]

    # 创建一个等长的观测目标位置的列表
    target_point_tuple = tuple(center)
    target_pos_list = [target_point_tuple] * len(camera_pos_list)

    return camera_pos_list, uav_pos_list, target_pos_list


def calculate_airborne_LiDAR_waypoints(start_point: List,
                                       end_point: List,
                                       flying_altitude: float,
                                       strip_overlap: float,
                                       lidar_fov: float):
    """
    函数 1: 机载LiDAR航线几何规划。恒定高度飞行模式
    返回: List[Tuple[Gf.Vec3d, Gf.Vec3d]] -> [(起点1, 终点1), (起点2, 终点2), ...]
    """
    # --- 核心物理参数计算 ---
    fov_rad = math.radians(lidar_fov)
    swath_width = 2 * flying_altitude * math.tan(fov_rad / 2)

    # 根据重叠度计算航线间隔 (Interval)
    interval = swath_width * (1.0 - strip_overlap / 100.0)

    # 计算需要飞行的总长度和航线数量
    # x_dist = abs(end_x - start_x) # 此处计算逻辑保留在应用层，因为计算函数只关心坐标点
    y_dist = abs(end_point[1] - start_point[1])
    flight_number = math.ceil(y_dist / interval) + 1

    waypoints_pairs = []

    for i in range(flight_number):
        y_pos = start_point[1] + i * interval

        # 实现 S 形往返飞行逻辑
        if i % 2 == 0:
            p1 = (start_point[0], y_pos, flying_altitude)
            p2 = (end_point[0], y_pos, flying_altitude)
        else:
            p1 = (end_point[0], y_pos, flying_altitude)
            p2 = (start_point[0], y_pos, flying_altitude)

        waypoints_pairs.append((p1, p2))

    return waypoints_pairs


# 测试模块
if __name__ == "__main__":
    # ------------ calculate_constant_altitude_sampling_waypoints恒定高度采用使用示例 --------------------------
    # # 模拟参数
    # test_altitude = 20.0
    # test_pixels = [500, 500]
    # test_focal = 20
    # test_h_aperture = 36

    # # 1. 测试独立计算 Footprint
    # gw, gh = calculate_footprint_from_camera(test_focal, test_h_aperture, test_pixels, test_altitude)
    # print("--- 独立Footprint测试 ---")
    # print(f"高度 {test_altitude}m 时，地面覆盖为: {gw:.2f}m x {gh:.2f}m\n")

    # # 2. 测试航点生成
    # start = (0, 0)
    # end = (50, 50)
    # overlap = [10, 10]  # 10% 重叠

    # pts, steps = calculate_constant_altitude_sampling_waypoints(
    #     start, end, test_altitude, test_pixels, test_focal, test_h_aperture, overlap
    # )

    # print("--- 航点生成测试 ---")
    # print(f"生成的航点总数: {len(pts)}")
    # print(f"前5个航点: {pts[:5]}")

    # ------------ calculate_semi_circular_sampling_waypoints半圆采样使用示例 --------------------------
    height = 10.0
    azimuth = 45.0  # 绕Z轴转45度
    cam_pts, uav_pts = calculate_semi_circular_sampling_waypoints(height, azimuth)

    print(f"采样点数量: {len(cam_pts)}")
    print(f"第一个采样点: {cam_pts[0]}")
