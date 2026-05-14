import numpy as np
import laspy
import os
from rapid.Utility import project_validity_check  # 项目有效性检查


def npy_to_las(npy_file_path):
    '''将配对的 pcd(局部坐标) 和 mat(变换矩阵) 转换为世界坐标系的 LAS 文件'''

    # 1. 筛选出所有的点云文件 (pcd_frame_xxx.npy)
    all_files = os.listdir(npy_file_path)
    pcd_files = [f for f in all_files if f.startswith('RTXLidarScanBuffer_') and f.endswith('.npy')]

    if not pcd_files:
        print("未找到 pcd_ 开头的点云文件，请检查输入路径。")
        return

    pcd_files.sort()  # 确保按帧顺序处理
    world_points_list = []

    print(f"正在处理 {len(pcd_files)} 帧数据并进行世界坐标变换...")

    # 2. 循环处理每一帧
    for pcd_name in pcd_files:
        # 获取帧 ID (假设格式为 pcd_frame_00000001.npy)
        frame_id = pcd_name.replace('RTXLidarScanBuffer_', '').replace('.npy', '')
        mat_name = f"WorldMatrix_{frame_id}.npy"

        pcd_full_path = os.path.join(npy_file_path, pcd_name)
        mat_full_path = os.path.join(npy_file_path, mat_name)

        # 检查矩阵文件是否存在
        if not os.path.exists(mat_full_path):
            print(f"警告: 找不到对应的矩阵文件 {mat_name}，跳过该帧。")
            continue

        # 加载数据
        local_points = np.load(pcd_full_path)  # (N, 3)
        world_matrix = np.load(mat_full_path)  # (4, 4)

        if local_points.shape[0] == 0:
            continue  # 无数据
        if local_points.ndim == 1:
            continue  # 单个点，数据形状(3,)，一个维度
        # 再次检查形状
        if world_matrix.shape != (4, 4):
            # 如果还是 12 行，取最后 4x4
            world_matrix = world_matrix.reshape(-1, 4)[-4:, :]
        # --- 核心步骤：局部坐标转世界坐标 ---
        # a. 构造齐次坐标 (N, 4)
        ones = np.ones((local_points.shape[0], 1))
        points_homo = np.hstack([local_points, ones])

        # b. 矩阵相乘 (N, 4) @ (4, 4) = (N, 4)
        # 注意：这里的矩阵在写入时已经根据需要做了转置，直接右乘即可
        transformed_points = points_homo @ world_matrix

        # c. 取前三列 (X, Y, Z)
        world_points = transformed_points[:, :3]
        world_points_list.append(world_points)

    if not world_points_list:
        print("没有有效的点云数据可以转换。")
        return

    # 3. 合并所有帧的数据
    all_world_data = np.vstack(world_points_list)
    print(f"合并完成。总点数shape: {all_world_data.shape}")

    # 4. 创建 LAS 文件
    header = laspy.LasHeader(point_format=6, version="1.4")

    # 自动计算偏移量（Offset）可以提高大规模场景下的存储精度
    header.offsets = np.min(all_world_data, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # 毫米级精度通常足够

    las = laspy.LasData(header)
    las.x = all_world_data[:, 0]
    las.y = all_world_data[:, 1]
    las.z = all_world_data[:, 2]

    # 5. 保存文件
    output_name = 'RTX_Lidar_World_Data.las'
    result_path = project_validity_check.get_folder("result")
    output_path = os.path.join(result_path, output_name)
    las.write(output_path)

    print(f"成功保存世界坐标点云至: {output_path}")


# 测试
if __name__ == "__main__":
    # 请根据实际路径修改
    test_path = r'E:\Downloads\isaac-sim-standalone-5.1.0\extsUser\rapid.LiDAR\rapid\result\PCD'
    if os.path.exists(test_path):
        npy_to_las(test_path)
    else:
        print("测试路径不存在。")