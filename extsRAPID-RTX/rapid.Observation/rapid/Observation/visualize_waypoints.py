import numpy as np
import carb
import math
from isaacsim.util.debug_draw import _debug_draw


def visualize_waypoints(waypoints, sensor_type='Perspective', look_at_target=None, view_length=5.0):
    """
    可视化航点、航线、航道箭头。
    如果是光学相机，还会可视化相机的观测方向(指向 look_at_target)。

    参数:
        waypoints: [(x, y, z), ...]
        sensor_type: 传感器类型字符串
        view_length: 相机观测方向线的长度 (米)
        look_at_target: [x, y, z] 观测中心点坐标。如果为 None,则默认向下看。
    """
    # 1. 获取接口
    draw = _debug_draw.acquire_debug_draw_interface()

    # 清理所有内容
    draw.clear_points()
    draw.clear_lines()

    if not waypoints or len(waypoints) < 1:
        return

    # 判定是否为雷达类型
    is_lidar = sensor_type in ["Airborne LiDAR", "Terrestrial LiDAR"]

    # 2. 颜色配置 (RGBA)
    color_path_line = (1, 1, 1, 0.4)    # 白色 (航线)
    color_path_arrow = (0, 1, 0, 1)    # 绿色 (航向)
    color_waypoint = (0, 0.5, 1, 1)    # 蓝色 (点)
    color_start = (1, 0, 0, 1)         # 红色 (起点)
    color_cam_view = (0, 1, 1, 1)      # 黄色 (相机观测向)

    pts = [tuple(p) for p in waypoints]
    num_pts = len(pts)

    # 3. 绘制基础航点
    draw.draw_points(pts, [color_waypoint] * num_pts, [10.0] * num_pts)
    draw.draw_points([pts[0]], [color_start], [25.0])  # 起点加粗

    # --- 容器：批量绘制线段 ---
    path_starts, path_ends, path_colors = [], [], []
    view_starts, view_ends, view_colors = [], [], []

    # 4. 遍历航点计算线段
    for i in range(num_pts):
        curr_p = np.array(pts[i])

        # --- A. 计算相机观测方向 (仅在非雷达模式下执行) ---
        if not is_lidar:
            # --- 核心修改部分：计算视线向量 ---
            if look_at_target is not None:
                # 计算指向中心的单位向量
                target_vec = np.array(look_at_target) - curr_p
                dist = np.linalg.norm(target_vec)
                if dist > 1e-6:
                    look_dir = target_vec / dist
                else:
                    look_dir = np.array([0, 0, -1])
            else:
                # 默认向下看
                look_dir = np.array([0, 0, -1])

            # 视线主线段终点
            target_p = curr_p + look_dir * view_length

            # 将主干加入容器
            view_starts.append(tuple(curr_p))
            view_ends.append(tuple(target_p))
            view_colors.append(color_cam_view)

            # --- 动态计算 V 字形箭头 (为了让箭头始终垂直于视线) ---
            # 找到一个垂直于 look_dir 的向量作为 side_v
            # 如果 look_dir 几乎垂直，则选择不同的基准向量以避免叉积为零
            up_ref = np.array([0, 0, 1]) if abs(look_dir[2]) < 0.9 else np.array([1, 0, 0])
            side_v = np.cross(look_dir, up_ref)
            side_v = (side_v / np.linalg.norm(side_v)) * 0.5  # 缩放到 0.5 米宽

            # 计算 V 字形的两个翼尖 (从终点向回拉一点并向两侧展开)
            wing_back = look_dir * 1.0  # 向回退 1 米
            wing1 = target_p - wing_back + side_v
            wing2 = target_p - wing_back - side_v

            view_starts.extend([tuple(target_p), tuple(target_p)])
            view_ends.extend([tuple(wing1), tuple(wing2)])
            view_colors.extend([color_cam_view, color_cam_view])

        # --- B. 计算航线及航向箭头 (保持不变) ---
        if i < num_pts - 1:
            next_p = np.array(pts[i+1])
            path_starts.append(tuple(curr_p))
            path_ends.append(tuple(next_p))
            path_colors.append(color_path_line)

            # 航向箭头逻辑
            mid_p = (curr_p + next_p) / 2
            fwd_dir = next_p - curr_p
            dist_fwd = np.linalg.norm(fwd_dir)
            if dist_fwd > 0.5:
                fwd_unit = fwd_dir / dist_fwd
                # 计算水平侧向向量用于画箭头翼
                side_p = np.array([-fwd_unit[1], fwd_unit[0], 0]) * 0.6
                p_wing1 = mid_p - fwd_unit * 1.2 + side_p
                p_wing2 = mid_p - fwd_unit * 1.2 - side_p

                path_starts.extend([tuple(mid_p), tuple(mid_p)])
                path_ends.extend([tuple(p_wing1), tuple(p_wing2)])
                path_colors.extend([color_path_arrow, color_path_arrow])

    # 5. 执行批量绘制
    if path_starts:
        draw.draw_lines(path_starts, path_ends, path_colors, [5] * len(path_starts))

    if not is_lidar and view_starts:
        draw.draw_lines(view_starts, view_ends, view_colors, [5] * len(view_starts))


def spaceborne_LiDAR_visualize(footprint_center, footprint_width, bottom_point=None, num_points=180, n=2.0, m=100.0):
    """
    参数:
        footprint_center: [x, y] 坐标
        footprint_width: 宽度
        bottom_point (np.array): [[x1, y1, z1], ...]可视化光斑的圆柱体底部的点
        num_points: 每一圈点的数量
        n: 每一层之间的垂直间距 (步长)
        m: 激光束到达的指定总高度 (目标高度)
    """
    # 1. 获取 debug draw 接口
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()

    # 2. 基本参数设置
    # 注意：radius = footprint_width * 2.0 会让圆圈非常大，
    # 如果想按标准直径显示，建议改为 footprint_width / 2.0
    radius = footprint_width / 2.0
    cx, cy = footprint_center       # 解析输入的 [x, y]

    all_points = []

    # 3. 计算需要生成的层数
    # 如果 m=50, n=2, 则生成 26 层 (从 0 到 50，包含起点和终点)
    num_layers = int(m / n) + 1

    # 4. 生成点阵
    for j in range(num_layers):
        current_z = j * n
        # 确保最后一层不会因为浮点误差显著超过目标高度 m
        if current_z > m:
            break

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points

            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            z = current_z

            all_points.append((float(x), float(y), float(z)))

    # 5. 颜色配置 (RGBA)
    color_cyan = (0, 1, 1, 0.8)  # 青色，带一点透明感

    # 6. 执行绘制
    if all_points:
        # 批量绘制圆柱体所有点
        draw.draw_points(all_points, [color_cyan] * len(all_points), [10.0] * len(all_points))
    if bottom_point:
        # 绘制底面上的点
        draw.draw_points(bottom_point, [color_cyan] * len(bottom_point), [10.0] * len(bottom_point))

    return all_points


if __name__ == "__main__":
    # --- 测试运行 ---
    # 创建一段模拟航线
    test_waypoints = []
    for i in range(5):
        x = i * 15
        for j in range(5):
            y = j * 15 if i % 2 == 0 else (4 - j) * 15
            test_waypoints.append((x, y, 30.0))

    visualize_waypoints(test_waypoints, view_length=8.0)
