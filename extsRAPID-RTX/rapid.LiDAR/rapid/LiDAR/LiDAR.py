# omni模块
from pxr import Gf, UsdGeom, Usd, Sdf
import omni.kit.commands
import omni.timeline
import omni.kit.app
from isaacsim.util.debug_draw import _debug_draw
import omni.replicator.core as rep
import omni.kit.notification_manager as nm
# 常用模块
import asyncio
import numpy as np
import math
from pathlib import Path
from typing import Dict, List

# 自定义模块
from .LiDAR_Writer import RTX_LiDARWriter
from .npy2las import npy_to_las
from rapid.Utility.sensor_params import get_prim_attributes
from rapid.Utility.calculate_sampling_waypoints import calculate_airborne_LiDAR_waypoints  # 计算航线位置
from rapid.Utility import project_validity_check  # 项目有效性检查
from rapid.Utility.simulation_progress_window import SimulationProgressWindow  # 进度条窗口
from .spaceborne_LiDAR import SpaceborneLiDARSimulation


class RTXLiDAR:
    def __init__(self):
        self.airborne_lidar_asset_path = str(Path(__file__).parent.parent.parent/'data'/'Airborne_lidar.usda')
        self.terrestrial_lidar_asset_path = str(Path(__file__).parent.parent.parent/'data'/'Terrestrial_lidar.usda')

    def main(self, simulation_parameters, visualize: bool = True):
        """
        机载和地基的模拟采用RTX_LiDAR模块功能,星载大光斑采用光线投射的命中点转波形
        参数:
        simulation_parameters (Dict): 仿真参数字典

        返回:
        """
        # 舞台初始化
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.time_codes_per_sec = self.timeline.get_time_codes_per_seconds()  # 获取stage帧率
        self.writer = None
        self.visualisation_writer = None
        self.render_product = None
        self.render_product_toVisualisation = None
        # 获取数据输出文件夹的路径
        intermediate_path = project_validity_check.get_folder("intermediate_data")
        self.intermediate_LiDAR_path = str(Path(intermediate_path) / "LiDAR")
        self.result_path = project_validity_check.get_folder("result")

        # LiDAR可视化
        self.visualize = visualize

        # 解析模拟参数
        self._parse_simulation_parameters(simulation_parameters)

        # -------------------------------初始化进度条窗口和可视化航点-----------------------
        self._progress_win = SimulationProgressWindow()
        self.draw = _debug_draw.acquire_debug_draw_interface()

        # 选择不同的模拟方式
        if self.params['sensor_type'] in ['Terrestrial LiDAR', 'Airborne LiDAR']:
            asyncio.ensure_future(self.run_simulation())
        elif self.params['sensor_type'] in ["Spaceborne LiDAR (Waveform LiDAR Data)"]:
            asyncio.ensure_future(SpaceborneLiDARSimulation.spaceborne_LiDAR_main(self.params, self._progress_win))

    async def run_simulation(self):
        '''
        '''
        try:
            total_time_codes = await self.simulation_core()
        except Exception as e:
            if self._progress_win:
                self._progress_win.status_label.text = f"Error: {str(e)}"
            raise
        finally:
            # 进度条窗口变化成Simulation Finished!
            if self._progress_win:
                self._progress_win.status_label.text = "Simulation Finished!"  # 窗口上的内容
                self._progress_win.close_btn.text = "Close"  # 右下角的按钮的内容
                self._progress_win.update_progress(total_time_codes)  # 进度条拉满
                # 2秒后自动隐藏窗口
                # await asyncio.sleep(2)
                # self._progress_win.visible = False

            # ----------------------------模拟结束后释放进程/显存与后处理-----------------------
            self.timeline.stop()
            # 资源清理，必须先断开writer在断开render_product
            self.writer.detach()
            if self.visualize:
                self.visualisation_writer.detach()
            # 让引擎处理一下断开writer连接的逻辑后断开render_product
            await omni.kit.app.get_app().next_update_async()
            self.render_product.destroy()
            if self.visualize:
                self.render_product_toVisualisation.destroy()

            # npy缓存转成las格式点云
            npy_to_las(self.intermediate_LiDAR_path)

    async def simulation_core(self):
        '''
        '''

        # 创建 LiDAR
        xformable = None
        total_time_codes, xformable = self._create_lidar()

        # RTX LiDAR的分辨率对于点云来说通常设为 [1, 1] 即可
        # 注意数据输出和可视化的Render Product的必须都在所有的Writer链接之前。
        self.render_product = rep.create.render_product(self.lidar_sensor_path, [1, 1])

        # 点云可视化
        if self.visualize:
            self._pointcloud_visualization()

        # 初始化写入器和链接渲染产品
        self.writer = rep.WriterRegistry.get("RTX_LiDARWriter")
        self.writer.initialize(
            output_dir=self.intermediate_LiDAR_path,
            lidar_path=self.lidar_sensor_path,
            RTX_LiDAR=True)
        self.writer.attach(self.render_product)

        # 进度条窗口总数计算
        self._progress_win.status_label.text = "Run Simulation..."  # 窗口上的内容
        self._progress_win.total_steps = total_time_codes

        # 开始模拟
        self.timeline.play()
        # 数据采集循环
        i = 0
        while i < total_time_codes:
            # 等待下一帧渲染完成
            await omni.kit.app.get_app().next_update_async()

            if self.params.get("sensor_type") == 'Airborne LiDAR':
                # 获取当前时间戳，计算当前帧的世界变换矩阵并提取平移向量 (Translation)
                current_time = Usd.TimeCode(self.timeline.get_current_time() * self.time_codes_per_sec)
                world_transform = xformable.ComputeLocalToWorldTransform(current_time)
                world_pos = world_transform.ExtractTranslation()
                curr_p = (world_pos[0], world_pos[1], world_pos[2])
                # 可视化当前位置
                self.draw.draw_points([curr_p], [(1, 0, 0, 1)], [25])

            # 增加进度条
            self._progress_win.update_progress(i)
            i += 1
        return total_time_codes

    def _parse_simulation_parameters(self, simulation_parameters: Dict = {}):
        '''解析读取仿真参数
        参数:
            simulation_complete (dict):数据格式如下{
            'Observation': {'sensor_stage_path': '/World/sensor'}
            },对于LiDAR数据并没有

        '''
        # ----------------------通过事件流数据获取对应的传感器--------------------------
        simulation_data = simulation_parameters['Observation']
        self.lidar_carrier_path = simulation_data['sensor_stage_path']
        self.lidar_carrier_prim = self.stage.GetPrimAtPath(self.lidar_carrier_path)
        if not self.lidar_carrier_prim.IsValid():
            # 如果不存在则直接返回
            nm.post_notification(
                "Sensor is not created. Please create a sensor before starting simulation.",
                status=nm.NotificationStatus.WARNING,
                duration=5)
            return False

        # ----------------------通过参数配置表获取传感器参数--------------------------
        self.params = {}
        # 首先读取基础参数 (Base_Sensor)
        get_prim_attributes(self.lidar_carrier_prim, "Base Attribute", self.params)
        # 读取光学传感器参数
        sensor_type = self.params.get("sensor_type")
        if sensor_type:
            get_prim_attributes(self.lidar_carrier_prim, sensor_type, self.params)

    def _create_lidar(self):
        LiDAR_scan_model = self.params['sensor_type']
        # LiDAR扫描模式,加载不同的usda配置文件
        if LiDAR_scan_model == 'Terrestrial LiDAR':
            lidar_asset_path = self.airborne_lidar_asset_path
        if LiDAR_scan_model == 'Airborne LiDAR':
            lidar_asset_path = self.airborne_lidar_asset_path

        # 加载已经已经配置好的RTX LiDAR的usda文件
        self.lidar_sensor_path = Sdf.Path(self.lidar_carrier_path).AppendPath('LiDAR')
        lidar_sensor_prim = self.stage.DefinePrim(self.lidar_sensor_path)
        lidar_sensor_prim.GetReferences().AddReference(lidar_asset_path)
        # 测试功能:使用默认的配置文件创建 RTX LiDAR
        # _, sensor = omni.kit.commands.execute(
        #     "IsaacSensorCreateRtxLidar",
        #     path=self.lidar_path,
        #     parent=None,
        #     config="Example_Rotary",  # 或者 "Velodyne_VLS128"
        #     translation=Gf.Vec3d(0.0, 0.0, 5.0),  # 抬高一点防止埋在地下
        #     orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        # )

        # 计算LiDAR属性set_LiDAR_attributes
        if LiDAR_scan_model == 'Terrestrial LiDAR':
            zenith_angle_resolution = self.params['terrestrial_LiDAR_zenith_angle_resolution']
            min_zenith_angle = self.params['terrestrial_LiDAR_min_zenith_angle']
            max_zenith_angle = self.params['terrestrial_LiDAR_max_zenith_angle']
            azimuth_angle_resolution = self.params['terrestrial_LiDAR_azimuth_angle_resolution']
            sampling_frequency = self.params['terrestrial_LiDAR_sampling_frequency']
            position = self.params['terrestrial_LiDAR_position']
            total_time_codes, xformable = self._set_terrestrial_LiDAR_attributes(
                lidar_sensor_prim, zenith_angle_resolution, min_zenith_angle,
                max_zenith_angle, azimuth_angle_resolution, sampling_frequency, position)

        # Airborne LiDAR设定
        if LiDAR_scan_model == 'Airborne LiDAR':
            # --------------------------设定lidar_sensor的属性--------------------------
            FOV = self.params['airborne_LiDAR_fov']
            angle_resolution = self.params['airborne_LiDAR_angle_resolution']
            scan_rate = self.params['airborne_LiDAR_scan_rate']
            self._set_airborne_LiDAR_attributes(lidar_sensor_prim, FOV, angle_resolution, scan_rate)

            # --------------------------设定载体移动属性--------------------------
            start_point = self.params['airborne_LiDAR_start_point']
            end_point = self.params['airborne_LiDAR_end_point']
            flight_altitude = self.params['airborne_LiDAR_flight_altitude']
            flight_speed = self.params['airborne_LiDAR_flight_speed']
            strip_overlap = self.params['airborne_LiDAR_strip_overlap']
            total_time_codes, xformable = self._control_airborne_carrier(
                start_point, end_point, flight_altitude,
                flight_speed, strip_overlap, FOV)

        return total_time_codes, xformable

    def _set_terrestrial_LiDAR_attributes(self,
                                          lidar_prim,
                                          vertical_angle_resolution: float = 0.1,
                                          vertical_start_angle: float = 30,
                                          vertical_end_angle: float = 150,
                                          horizontal_angle_resolution: float = 0.1,
                                          sampling_frequency: float = 100000,
                                          position: List = [0, 0, 10],
                                          horizontal_start_angle: float = 0,
                                          horizontal_end_angle: float = 360,):
        '''配置地基LiDAR文件
        第一、输入控制(角度分辨率、角度范围)参数来产生配置参数(reportRateBaseHz、elevationDeg等),输出扫描时间（单位秒）
        第二、 这种方法中,控制LiDAR一圈只tick一次, 一次tick全部的垂直方位角,以载体的旋转控制水平方位角
        载体每圈的tick数=360/方位角分辨率, 载体Rotate一圈的时间=扫秒的总点数（）/采样频率,
        载体的tick频率reportRateBaseHz=每圈的tick数/ Rotate一圈的时间.
        而LiDAR的scanRateBaseHz没有了意义,配合LiDAR的reportRateBaseHz实现一圈一次tick,所以scanRateBaseHz=reportRateBaseHz
        '''
        # --------------------------计算属性--------------------------
        # --- 1. 坐标系转换 (天顶角 -> LiDAR内置角度) ---
        lidar_start = 90.0 - vertical_start_angle
        lidar_end = 90.0 - vertical_end_angle

        # --- 2. 计算点数和步长方向 ---
        # 计算总行程跨度
        angle_range = abs(lidar_end - lidar_start)

        # 计算理论点数 (使用abs确保点数为正)
        # +1 是为了确保包含终点
        vertical_angle_number = int(abs(angle_range) / vertical_angle_resolution) + 1

        # --- 3. 产生角度数组 ---
        # 推荐使用 np.linspace，它能完美处理起始/终点的包含关系，且自动处理步长正负
        vertical_angle = np.linspace(lidar_start, lidar_end, vertical_angle_number).tolist()

        # 水平角度数量=载体每圈tick数量
        # horizontal_angle = np.arange(horizontal_end_angle, horizontal_start_angle+horizontal_angle_resolution, horizontal_angle_resolution).tolist()
        tick_number_per_scan = (abs(horizontal_start_angle) +
                                abs(horizontal_end_angle))/horizontal_angle_resolution
        # 扫描时间参数
        total_scanning_time = (vertical_angle_number*tick_number_per_scan)/sampling_frequency  # Rotate一圈的时间
        reportRateBaseHz = tick_number_per_scan/total_scanning_time
        scanRateBaseHz = reportRateBaseHz

        # --------------------------设定LiDAR属性--------------------------
        lidar_prim.GetAttribute("omni:sensor:Core:validStartAzimuthDeg").Set(int(horizontal_start_angle))
        lidar_prim.GetAttribute("omni:sensor:Core:validEndAzimuthDeg").Set(int(horizontal_end_angle))
        lidar_prim.GetAttribute("omni:sensor:Core:reportRateBaseHz").Set(int(reportRateBaseHz))
        lidar_prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Set(int(scanRateBaseHz))
        lidar_prim.GetAttribute("omni:sensor:Core:numberOfEmitters").Set(int(vertical_angle_number))
        lidar_prim.GetAttribute("omni:sensor:Core:numberOfChannels").Set(int(vertical_angle_number))
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:channelId").Set([i for i in range(1, vertical_angle_number+1)])
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:azimuthDeg").Set([0 for _ in range(vertical_angle_number)])
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:elevationDeg").Set(vertical_angle)
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:fireTimeNs").Set([0 for _ in range(vertical_angle_number)])

        # --------------------------设定载体旋转属性--------------------------
        # 设定载体位置
        self.lidar_carrier_prim.GetAttribute('xformOp:translate').Set((position[0], position[1], position[2]))
        # 转动RTX_lidar载体
        xformable = UsdGeom.Xformable(self.lidar_carrier_prim)
        total_time_codes = float(total_scanning_time*self.time_codes_per_sec)

        OrientOp = self.get_or_add_rotate_op(xformable)
        OrientOp.Set(time=0, value=(0, 0, horizontal_start_angle))
        OrientOp.Set(time=total_time_codes, value=(0, 0, horizontal_end_angle))
        return total_time_codes, xformable

    def get_or_add_rotate_op(self, xformable):
        # 1. 获取当前所有已排序的变换操作 (Translate, Rotate, Scale 等)
        ordered_ops = xformable.GetOrderedXformOps()

        # 2. 遍历查找类型为 RotateXYZ 的操作
        orient_op = None
        for op in ordered_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                orient_op = op
                break

        # 3. 如果没找到，则创建一个新的
        if not orient_op:
            print("未找到 RotateXYZ，正在新建...")
            orient_op = xformable.AddRotateXYZOp()
        else:
            print("找到已存在的 RotateXYZ，直接获取。")

        return orient_op

    def _set_airborne_LiDAR_attributes(self,
                                       lidar_prim,
                                       FOV_angle: float = 80,
                                       horizontal_angle_resolution: float = 0.06,
                                       scanRateBaseHz: float = 60,):
        # --------------------------计算属性--------------------------
        # 计算起始点和reportRateBaseHz
        start_angle = FOV_angle/2
        reportRateBaseHz = (FOV_angle/horizontal_angle_resolution)*scanRateBaseHz

        # 读写LiDAR配置文件
        # --------------------------设定LiDAR属性--------------------------
        lidar_prim.GetAttribute("omni:sensor:Core:validStartAzimuthDeg").Set(360-start_angle)
        lidar_prim.GetAttribute("omni:sensor:Core:validEndAzimuthDeg").Set(start_angle)
        lidar_prim.GetAttribute("omni:sensor:Core:reportRateBaseHz").Set(int(reportRateBaseHz))
        lidar_prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Set(int(scanRateBaseHz))
        lidar_prim.GetAttribute("omni:sensor:Core:numberOfEmitters").Set(1)
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:azimuthDeg").Set([0 for i in range(1)])
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:elevationDeg").Set([0 for i in range(1)])
        lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:fireTimeNs").Set([0 for i in range(1)])

    def _control_airborne_carrier(self,
                                  start_point: List,
                                  end_point: List,
                                  flying_altitude: float = 50,
                                  carrier_ground_speed: float = 10,
                                  strip_overlap: float = 20,
                                  lidar_fov: float = 60):
        '''
        机载载体控制
        '''
        waypoints_pairs = calculate_airborne_LiDAR_waypoints(start_point, end_point, flying_altitude, strip_overlap, lidar_fov)

        # 检查xform
        xformable = UsdGeom.Xformable(self.lidar_carrier_prim)

        # 设置旋转位姿 (保持不变)
        quat = self.euler2quat(0, 90, 0)
        self.lidar_carrier_prim.GetAttribute('xformOp:orient').Set(quat)

        # 确保有 translate 操作符
        translateOp = xformable.GetTranslateOp()
        if not translateOp:
            translateOp = xformable.AddTranslateOp()

        current_time = 0.0

        for i, (p1, p2) in enumerate(waypoints_pairs):
            p1 = Gf.Vec3d(p1)
            p2 = Gf.Vec3d(p2)
            # 计算当前航线的位移距离
            dist = (p2 - p1).GetLength()
            # 计算该段航线飞行所需的 TimeCodes (秒 * TimeCodes/Sec)
            line_duration_tc = (dist / carrier_ground_speed) * self.time_codes_per_sec

            # 1. 设置航线起点
            translateOp.Set(time=current_time, value=p1)

            # 2. 增加飞行时间
            current_time += line_duration_tc

            # 3. 设置航线终点
            translateOp.Set(time=current_time, value=p2)

            # 4. 增加航线间的跳跃缓冲 (1.0 TimeCode)
            # 只有不是最后一条航线时才增加缓冲，防止最后多出一帧空白
            if i < len(waypoints_pairs) - 1:
                current_time += 1.0

        return int(current_time), xformable

    def euler2quat(sell, phi, theta, psi):
        # 将欧拉角转换为弧度
        phi_rad = math.radians(phi)
        theta_rad = math.radians(theta)
        psi_rad = math.radians(psi)

        # 计算半角的余弦和正弦值
        c1 = math.cos(phi_rad / 2)
        c2 = math.cos(theta_rad / 2)
        c3 = math.cos(psi_rad / 2)
        s1 = math.sin(phi_rad / 2)
        s2 = math.sin(theta_rad / 2)
        s3 = math.sin(psi_rad / 2)

        # 计算四元数的四个分量
        quat = Gf.Quatd(
            c1 * c2 * c3 + s1 * s2 * s3,  # w
            s1 * c2 * c3 - c1 * s2 * s3,  # x
            c1 * s2 * c3 + s1 * c2 * s3,  # y
            c1 * c2 * s3 - s1 * s2 * c3   # z
        )

        return quat

    def _pointcloud_visualization(self):
        '''点云可视化, self.render_product_toVisualisation必须在
        '''
        # 创建用于可视化的 Render Product
        self.render_product_toVisualisation = rep.create.render_product(self.lidar_sensor_path, [1, 1])
        # Create a Replicator Writer that "writes" points into the scene for debug viewing
        self.visualisation_writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        self.visualisation_writer.initialize(size=0.5)  # 设定点的大小
        self.visualisation_writer.attach(self.render_product_toVisualisation)
