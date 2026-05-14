import asyncio
import laspy
import numpy as np
from pathlib import Path
import omni.kit.raycast.query
import matplotlib.pyplot as plt
# 自定义模块
from rapid.Utility import project_validity_check  # 项目有效性检查


class SpaceborneLiDARSimulation:
    @staticmethod
    async def spaceborne_LiDAR_main(data, progress_win):
        """
        综合函数:生成采样点、射线检测、获取Z轴范围
        Returns:
            points_3d: (N, 6) 数组 [x, y, z, nx, ny, nz]
            z_range: [z_min, z_max]
        """
        # 简单进度条窗口
        progress_win.status_label.text = "Run Spaceborne LiDAR Simulation..."  # 窗口上的内容
        progress_win.total_steps = 100

        # 获取参数
        result_path = project_validity_check.get_folder("result")
        footprint_center = data["spaceborne_LiDAR_footprint_center"]
        footprint_width = data["spaceborne_LiDAR_footprint_width"]
        system_pulse_width = data["spaceborne_LiDAR_system_pulse_width"]
        vertical_bin_size = data["spaceborne_LiDAR_vertical_bin_size"]

        # --------------采样单个光斑中的所有命中点位置（点云模拟）-------------
        hit_points_and_normal, [z_min, z_max] = await SpaceborneLiDARSimulation.raycast(footprint_center, footprint_width)

        # --------------点云转波形信息--------------
        # 补全列: [x, y, z, intensity, return_number, n_hits]
        hit_points = hit_points_and_normal[:, :3]  # 去掉法线维，只保留position参数
        mock_data = np.zeros((hit_points.shape[0], 6))
        mock_data[:, :3] = hit_points
        mock_data[:, 5] = 1  # 假设所有点都是单次返回

        # 开始模拟波形
        z_axis, waveform, raw_profile = SpaceborneLiDARSimulation.simulate_waveform(
            mock_data, sigma_f=footprint_width, sigma_p=system_pulse_width, bin_res=vertical_bin_size, method='count')
        # 强度归一化 (将波形值缩放到 0-1)
        waveform, raw_profile = SpaceborneLiDARSimulation.normalize_waveform_intensity(waveform, raw_profile)

        # 计算相对高度RH指标, 画单个光斑的波形图
        plot_result_path = str(Path(result_path) / 'SpaceborneLiDAR_Waveform.png')
        rh_values = SpaceborneLiDARSimulation.calculate_rh_metrics(z_axis, waveform)
        SpaceborneLiDARSimulation.plot_wavefrom(z_axis, waveform, raw_profile, rh_values, plot_result_path)

        # --------------可视化和保存波形与点云数据--------------
        pointcloud_file_path = str(Path(result_path) / 'SpaceborneLiDAR_PointCloud.las')
        waveform_file_path = str(Path(result_path) / 'SpaceborneLiDAR_Waveform.txt')
        meta = {
            "center_x": footprint_center[0],
            "center_y": footprint_center[1],
            "width": footprint_width,
            "pulse_width": system_pulse_width,
            "bin_size": vertical_bin_size
            }
        SpaceborneLiDARSimulation.save_waveform_to_txt(z_axis, waveform, rh_values, meta, waveform_file_path)
        SpaceborneLiDARSimulation.save_and_visualize_pointcloud(hit_points, pointcloud_file_path)

        # 进度条窗口更新
        progress_win.status_label.text = "Simulation Finished!"  # 窗口上的内容
        progress_win.close_btn.text = "Close"  # 右下角的按钮的内容
        progress_win.update_progress(100)  # 进度条拉满

    @staticmethod
    async def raycast(footprint_center, footprint_width, sampling_step=0.03):
        """
        综合函数:生成采样点、射线检测、获取Z轴范围
        Returns:
            points_3d: (N, 6) 数组 [x, y, z, nx, ny, nz]
            z_range: [z_min, z_max]
        """
        # 1. 采样LiDAR光斑圆内的 XY 点
        xy_samples = SpaceborneLiDARSimulation.sample_points_in_circle(footprint_center, footprint_width, sampling_step)
        # 2. 执行射线检测获取命中点
        points_3d = await SpaceborneLiDARSimulation.get_heights_from_xy(xy_samples)

        # 3. 计算 Z 轴高度范围
        z_values = points_3d[:, 2]
        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))

        return points_3d, [z_min, z_max]

    @staticmethod
    def sample_points_in_circle(center, diameter, step=0.1):
        """
        在圆范围内进行均匀网格采样
        Args:
            center: [x, y]
            diameter: 直径
            step: 采样步长 (米)，步长越小点越密
        Returns:
            np.array: [[x1, y1], [x2, y2], ...]
        """
        cx, cy = center
        radius = diameter / 2.0

        # 1. 在外接正方形内生成网格
        x_range = np.arange(cx - radius, cx + radius + step, step)
        y_range = np.arange(cy - radius, cy + radius + step, step)
        xv, yv = np.meshgrid(x_range, y_range)

        # 2. 扁平化并计算到中心的距离
        pts = np.vstack([xv.ravel(), yv.ravel()]).T
        dist = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)

        # 3. 仅保留圆内部的点
        return pts[dist <= radius]

    @staticmethod
    async def get_heights_from_xy(x_y_list):
        """
        静态函数：根据 XY 列表进行射线采样，获取高度和法线。
        Args:
            x_y_list: np.array 或 list, 格式为 [[x1, y1], [x2, y2], ...]
        Returns:
            np.array: 形状为 (N, 6) 的数组，列为 [x, y, z, nx, ny, nz]
        """
        num_rays = len(x_y_list)
        results_3d = np.zeros((num_rays, 6), dtype=np.float32)
        # 内部状态追踪器（使用字典以便在内部函数中修改）
        state = {"completed_count": 0}
        raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()

        # 定义内部回调函数
        def _on_hit_internal(ray, result, idx, x, y):
            if result.valid:
                results_3d[idx] = [
                    x, y, 
                    result.hit_position[2],
                    result.normal[0],
                    result.normal[1],
                    result.normal[2]
                ]
            else:
                # 未击中时的默认值（高度0，法线向上）
                results_3d[idx] = [x, y, 0.0, 0.0, 0.0, 1.0]
            state["completed_count"] += 1

        # 批量提交射线请求
        for i in range(num_rays):
            px, py = x_y_list[i]
            # 射线从高空 10000.0 垂直向下发射
            ray = omni.kit.raycast.query.Ray(
                (float(px), float(py), 10000.0),
                (0.0, 0.0, -1.0)
            )
            # 使用 lambda 闭包传递索引和原始坐标
            raycast_interface.submit_raycast_query(
                ray, 
                lambda r, res, idx=i, x=px, y=py: _on_hit_internal(r, res, idx, x, y)
            )
        # 异步等待所有采样完成
        while state["completed_count"] < num_rays:
            await asyncio.sleep(0.001)
        return results_3d

    @staticmethod
    def simulate_waveform(points, sigma_f, sigma_p, bin_res=0.15, method='count'):

        """
        复现 Hancock et al. (2019) 的波形模拟算法
        参数:
        points: 数组 [N, 6]，列分别为 [x, y, z, intensity, return_number, number_of_returns]
        x0, y0: 足迹中心坐标
        sigma_f: 足迹宽度 (Footprint width),可视化的光斑大小采用的是4sigma_f
        sigma_p: 系统脉冲宽度 (System pulse width)
        bin_res: 垂直分箱分辨率 (默认 0.15m，GEDI采样率)
        method: Ii 的计算方式 ('count', 'frac', 'int')
        """
        # 计算足迹中心坐标
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        # 移动点云到 0,0
        points[:, 0] -= center_x
        points[:, 1] -= center_y

        # 计算点到中心的水平距离平方
        dx = points[:, 0] - 0
        dy = points[:, 1] - 0
        dist_sq = dx**2 + dy**2

        # 2. 计算公式 (1): 足迹强度分布权重 (Gaussian Footprint Weighting)
        # Iw,i = (1 / (sigma_f * sqrt(2*pi))) * exp(-dist_sq / (2 * sigma_f^2))
        iw_spatial = (1.0 / (sigma_f * np.sqrt(2 * np.pi))) * np.exp(-dist_sq / (2 * sigma_f**2))

        # 3. 计算 Ii (局部点击权重)
        if method == 'count':
            ii = np.ones(len(points))
        elif method == 'frac':
            # Ii = 1 / nHits
            ii = 1.0 / points[:, 5]
        elif method == 'int':
            # Ii 正比于反射强度
            ii = points[:, 3]
        else:
            ii = np.ones(len(points))

        # 最终点的权重 Iw,i
        weights = iw_spatial * ii

        # 4. 垂直分箱 (Binning)
        # 确定垂直范围
        z_min = np.floor(np.min(points[:, 2])) - 5
        z_max = np.ceil(np.max(points[:, 2])) + 5
        bins = np.arange(z_min, z_max, bin_res)

        # 将点的权重累加到垂直 bin 中 (得到理想波形 I(z) 的离散表达)
        # 这里对应公式 (2) 的求和部分，但在卷积之前执行
        ideal_profile, _ = np.histogram(points[:, 2], bins=bins, weights=weights)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        # 5. 构建公式 (3): 系统脉冲 p(z) (Gaussian Pulse)
        # 脉冲窗口大小取 4*sigma_p
        p_z_range = np.arange(-4 * sigma_p, 4 * sigma_p, bin_res)
        pulse = (1.0 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-(p_z_range**2) / (2 * sigma_p**2))
        pulse /= np.sum(pulse)  # 归一化能量

        # 6. 执行卷积 (Convolution)
        # I(z) = Sum(Iw,i) ⊗ p(z)
        simulated_waveform = np.convolve(ideal_profile, pulse, mode='same')

        return bin_centers, simulated_waveform, ideal_profile

    @staticmethod
    def normalize_waveform_intensity(waveform, raw_profile):
        """
        将模拟波形和原始能量分布剖面的强度归一化到 0-1 之间
        """
        # 处理波形归一化
        wf_min = np.min(waveform)
        wf_max = np.max(waveform)
        if wf_max > wf_min:
            waveform = (waveform - wf_min) / (wf_max - wf_min)
        else:
            # 如果波形是平的，则全设为0
            waveform = np.zeros_like(waveform)

        # 处理原始能量剖面归一化 (确保背景填充和波形在同一量级)
        rp_min = np.min(raw_profile)
        rp_max = np.max(raw_profile)
        if rp_max > rp_min:
            raw_profile = (raw_profile - rp_min) / (rp_max - rp_min)
        else:
            raw_profile = np.zeros_like(raw_profile)

        return waveform, raw_profile

    @staticmethod
    def calculate_rh_metrics(z_axis, waveform, percentiles=[25, 50, 95]):
        """
        计算相对高度 (RH) 指标
        注意：根据 GEDI 标准,RH 是基于地面到信号顶部的累积能量计算的
        """
        # 1. 确保波形值为正数（减去可能的噪声均值）
        clean_waveform = np.maximum(waveform - np.min(waveform), 0)

        # 2. 计算累积能量 (从地面向上累积)
        # 注意：z_axis 如果是从低到高，直接 cumsum；如果是从高到低，需要翻转
        cumulative_energy = np.cumsum(clean_waveform)
        total_energy = cumulative_energy[-1]

        if total_energy <= 0:
            return {p: 0 for p in percentiles}

        relative_cumulative = cumulative_energy / total_energy

        # 3. 通过线性插值寻找对应百分比的高度
        rh_values = {}
        for p in percentiles:
            # 寻找能量比例达到 p% 的高度
            target = p / 100.0
            rh_height = np.interp(target, relative_cumulative, z_axis)
            rh_values[p] = rh_height

        return rh_values

    @staticmethod
    def plot_wavefrom(z_axis, waveform, raw_profile, rh_values, save_path):
        # ================= 字体参数统一控制区 =================
        TITLE_FONTSIZE = 20
        AXIS_LABEL_FONTSIZE = 20
        TICK_LABEL_FONTSIZE = 16
        LEGEND_FONTSIZE = 16
        RH_TEXT_FONTSIZE = 18
        # =====================================================

        plt.figure(figsize=(8, 10))

        # A. 原始能量分布
        plt.fill_betweenx(
            z_axis, 0, raw_profile,
            color='gray', alpha=0.2,
            label='Point Distribution'
        )

        # B. 模拟全波形
        plt.plot(
            waveform, z_axis,
            color='#1f77b4', linewidth=2.5,
            label='Simulated Waveform'
        )

        # C. RH 标记
        colors = {95: 'red', 50: 'green', 25: 'orange'}

        for p in [25, 50, 95]:
            h = rh_values[p]

            plt.hlines(
                y=h,
                xmin=0,
                xmax=np.max(waveform) * 1.05,
                colors=colors[p],
                linestyles='--',
                linewidth=1.5
            )

            plt.text(
                np.max(waveform) * 0.6,
                h + 0.3,
                f'RH{p}: {h:.2f} m',
                color=colors[p],
                fontsize=RH_TEXT_FONTSIZE,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

        # D. 图表装饰
        plt.title(
            'LiDAR Waveform Simulation with RH Metrics',
            fontsize=TITLE_FONTSIZE,
            pad=20
        )
        plt.xlabel('Intensity', fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel('Elevation (z) [m]', fontsize=AXIS_LABEL_FONTSIZE)

        plt.ylim(np.min(z_axis) - 2, np.max(z_axis) + 5)
        plt.xlim(0, np.max(waveform) * 1.1)

        # 坐标刻度字体大小
        plt.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_and_visualize_pointcloud(xyz_data, data_path):
        # -----------------------保存波形文件-----------------------

        # -----------------------保存点云文件-----------------------
        # 创建LAS文件头
        header = laspy.LasHeader(version="1.4", point_format=6)
        # 设置数据的尺度偏移（根据你的数据调整，这里假设单位为米）
        header.x_scale = 0.001
        header.y_scale = 0.001
        header.z_scale = 0.001
        header.x_offset = np.mean(xyz_data[:, 0])
        header.y_offset = np.mean(xyz_data[:, 1])
        header.z_offset = np.mean(xyz_data[:, 2])
        # 创建点云记录并写入数据
        las = laspy.LasData(header)
        las.x = xyz_data[:, 0]
        las.y = xyz_data[:, 1]
        las.z = xyz_data[:, 2]
        # 保存点云文件
        las.write(data_path.replace('.las', '_part.las'))

        # -----------------------点云可视化-----------------------

    @staticmethod
    def save_waveform_to_txt(z_axis, waveform, rh_values, metadata, save_path):
        """
        将波形数据和元数据保存为带有头信息的 TXT 文件

        参数:
            z_axis: 高度/距离轴数组
            waveform: 强度数组
            rh_values: 字典，如 {25: 10.5, 50: 15.2, 95: 22.1}
            metadata: 字典，包含 center, width, pulse_width 等
            save_path: 文件保存路径
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            # 写入基本参数
            f.write(f"# Footprint_Center_X: {metadata.get('center_x', 0):.4f}\n")
            f.write(f"# Footprint_Center_Y: {metadata.get('center_y', 0):.4f}\n")
            f.write(f"# Footprint_Width_m: {metadata.get('width', 25.0):.2f}\n")
            f.write(f"# System_Pulse_Width_m: {metadata.get('pulse_width', 10.0):.2f}\n")
            f.write(f"# Vertical_Bin_Size_m: {metadata.get('bin_size', 10.0):.2f}\n")


            # 写入关键指标 (RH值)
            f.write("# --- RH Metrics ---\n")
            for p in sorted(rh_values.keys()):
                f.write(f"# RH{p}: {rh_values[p]:.4f}\n")

            f.write("# ==================================================\n")

            # 2. 写入数据列标题
            f.write("Elevation_m\tIntensity_Normalized\n")

            # 3. 写入波形数据 (使用 Tab 分隔)
            # zip 将两个一维数组组合
            for z, i in zip(z_axis, waveform):
                f.write(f"{z:.4f}\t{i:.6f}\n")