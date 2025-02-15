import omni.kit.commands
from pxr import Gf, UsdGeom, Usd
import omni.replicator.core as rep
import numpy as np
import asyncio
from pathlib import Path
from .TestWriter import RTX_LidarWriter
import open3d as o3d
import os
import omni.timeline
import json
import math
# .pyd格式库引入时就是白色字体，这个包已经保留在扩展中
from omni.syntheticdata._syntheticdata import acquire_syntheticdata_interface

Terrestrial_Lidar_config = "Terrestrial_lidar"
Airborne_Lidar_config = 'Airborne_lidar'
Terrestrial_lidar_json_path = r'E:\Downloads\isaac-sim-4.5\exts\isaacsim.sensors.rtx\data\lidar_configs\RAPID\Terrestrial_lidar.json'
Airborne_lidar_json_path = r'E:\Downloads\isaac-sim-4.5\exts\isaacsim.sensors.rtx\data\lidar_configs\RAPID\Airborne_lidar.json'


# 将输出的所有npy数据转换为pcd数据,并保存
def np_to_pcd(path_output):
    '''将所有npy数据转换为pcd数据,并保存'''
    # 读取所有npy数据
    filename_list = [file for file in os.listdir(path_output) if file.endswith('.npy')]
    npy_data_shape = np.load(os.path.join(path_output, filename_list[0])).shape[1]  # 读取npy的列数
    all_npy_data = np.zeros((1, npy_data_shape))  # 定义一个和原数据同列数的空数组

    # 循环读取所有数据
    for filename in filename_list:
        new_data = np.load(os.path.join(path_output, filename))
        all_npy_data = np.concatenate((all_npy_data, new_data), axis=0)

    # 创建一个 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    # 将 NumPy 数组转换为 Open3D 点云的点
    pcd.points = o3d.utility.Vector3dVector(all_npy_data)
    # 变换绝对路径
    output_path_Bytes = os.path.abspath(path_output/'RTX_Lidar_Data.pcd')  # 二进制文件
    # output_path_ASCII = os.path.abspath(output_path/'RTX_Lidar_Data_ASCII.pcd')  # 文本可读文件
    # 保存点云对象为 .pcd 文件
    o3d.io.write_point_cloud(output_path_Bytes, pcd)
    # o3d.io.write_point_cloud(output_path_ASCII, pcd, write_ascii=True)


class rtx_lidar:
    def __init__(self):
        # 获取舞台
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        # 获取默认时间面板
        self.timeline = omni.timeline.get_timeline_interface()

        # 定义数据输出文件路径
        self.path_output = Path(__file__).parent.parent/'result/PCD'

        # 激光雷达参数（待定）

        # 可视化与数据输出的切换
        Visualisation = True

        # 异步输出，先输出.npy格式的点云后转换为pcd格式
        async def go():
            # 运行主程序
            self.Lidar_main(Visualisation)
            await rep.orchestrator.run_async()

            await rep.orchestrator.run_until_complete_async()
            # 将npy文件转化为pcd文件
            # np_to_pcd(self.path_output)
        asyncio.ensure_future(go())

    # 创建lidar，使用RTX_LidarWriter输出点云数据
    def Lidar_main(self, Visualisation: bool = True,
                   Lidar_scan_model: str = 'Airborne'):

        # 创建Lidar载体
        xformable = self.creat_xform()

        # lidar文件配置与扫描模式选择
        if Lidar_scan_model == 'Terrestrial':
            '''目前是只有单站扫描, Terrestrial_scan之后还需要添加起始与结束的扫描角度, 多站点等参数'''
            # end_time_seconds返回单站扫描的时间（秒）
            # 修改Lidar配置文件json的参数
            end_time_seconds = self.Terrestrial_Lidar_configuration()
            # 创建lidar
            sensor = self.creat_Liar(Terrestrial_Lidar_config, position=(0, 0, 3.0))
            # 控制扫描过程（扫描结束时间，end_time秒，end_time_codes帧数）
            end_time_codes = self.Terrestrial_scan(carrier=xformable[1],
                                                   end_time=end_time_seconds)
        if Lidar_scan_model == 'Airborne':
            ''''''
            self.Airborne_Lidar_configuration()
            print('1111111111111111111111111111111111111111111111111111111111111111')
            sensor = self.creat_Liar(Airborne_Lidar_config, position=(0, 0, 0))
            end_time_codes = self.Airborne_scan()

        # Create and Attach a render product to the camera
        render_product_toData = rep.create.render_product(sensor.GetPath(), [1, 1])

        # 可视化模块
        if Visualisation:
            self.Lidar_Visualisation(sensor)
            pass

        # 触发器设定，当到达最大扫描时间停止
        with rep.trigger.on_frame(num_frames=int(end_time_codes/5)+1):
            pass

        # Initialize and attach writer,数据输出
        writer = rep.writers.get("RTX_LidarWriter")
        writer.initialize(
            output_dir=self.path_output,
            RTX_Lidar=True,
            transformPoints=True)
        writer.attach(render_product_toData)

    # 创建Lidar载体
    def creat_xform(self):
        '''创建Lidar载体'''
        lidar_carrier_path = "/World/RTX_lidar"
        omni.kit.commands.execute('CreatePrimWithDefaultXform', prim_type='Xform', prim_path=lidar_carrier_path)
        lidar_carrier_prim = self.stage.GetPrimAtPath(lidar_carrier_path)
        xformable = UsdGeom.Xformable(lidar_carrier_prim)  # 添加Xformable移动功能
        xformable.ClearXformOpOrder()
        return lidar_carrier_prim, xformable

    # 配置地基Lidar文件
    def Terrestrial_Lidar_configuration(self, vertical_angle_resolution: float = 0.05,
                                        vertical_start_angle: float = 30,
                                        vertical_end_angle: float = -30,
                                        horizontal_angle_resolution: float = 0.05,
                                        horizontal_start_angle: float = 0,
                                        horizontal_end_angle: float = 360,
                                        sampling_frequency: float = 100000):
        '''配置地基Lidar文件
        第一、输入控制(角度分辨率、采样频率等)参数来产生配置参数(reportRateBaseHz、elevationDeg等),输出扫描时间（单位秒）
        第二、 这种方法中,控制Lidar一圈只tick一次, 一次tick全部的垂直方位角,以载体的旋转控制水平方位角
        载体每圈的tick数=360/方位角分辨率, 载体Rotate一圈的时间=扫秒的总点数（）/采样频率,
        载体的tick频率reportRateBaseHz=每圈的tick数/ Rotate一圈的时间.
        而Lidar的scanRateBaseHz没有了意义,配合Lidar的reportRateBaseHz实现一圈一次tick,所以scanRateBaseHz=reportRateBaseHz
        '''

        # 计算Lidar垂直角度和角度数量，载体每圈tick次数(旋转Lidar方案不需要设定水平角度，它由载体决定)
        vertical_angle = np.arange(vertical_end_angle, vertical_start_angle+vertical_angle_resolution,
                                   vertical_angle_resolution).tolist()
        vertical_angle_number = len(vertical_angle)
        # 水平角度数量=载体每圈tick数量
        # horizontal_angle = np.arange(horizontal_end_angle, horizontal_start_angle+horizontal_angle_resolution, horizontal_angle_resolution).tolist()
        tick_number_per_scan = (abs(horizontal_start_angle) +
                                abs(horizontal_end_angle))/horizontal_angle_resolution

        # 扫描时间参数
        time = (vertical_angle_number*tick_number_per_scan)/sampling_frequency  # Rotate一圈的时间
        reportRateBaseHz = tick_number_per_scan/time
        scanRateBaseHz = reportRateBaseHz

        # 多线扫描
        lines_number = 5
        azimuthDeg = []
        for i in range(lines_number):
            azimuthDeg.append([i for _ in range(len(vertical_angle))])

        # 读写Lidar配置文件
        with open(Terrestrial_lidar_json_path, 'r') as f:
            data = json.load(f)
        # 修改数据
        data['profile']['validStartAzimuthDeg'] = horizontal_start_angle
        data['profile']['validEndAzimuthDeg'] = horizontal_end_angle
        data['profile']['reportRateBaseHz'] = reportRateBaseHz
        data['profile']['scanRateBaseHz'] = scanRateBaseHz
        data['profile']['numberOfEmitters'] = len(vertical_angle)
        data['profile']['emitterStates'][0]['azimuthDeg'] = [0 for _ in range(len(vertical_angle))]
        data['profile']['emitterStates'][0]['elevationDeg'] = vertical_angle
        data['profile']['emitterStates'][0]['fireTimeNs'] = [0 for _ in range(len(vertical_angle))]
        # 将修改后的数据写回JSON文件
        with open(Terrestrial_lidar_json_path, 'w') as f:
            json.dump(data, f, indent=3)

        # 返回完成一圈扫描的时间
        return time

    # 配置机载Lidar文件
    def Airborne_Lidar_configuration(self,
                                     FOV_angle: float = 120,
                                     horizontal_angle_resolution: float = 0.005,
                                     scanRateBaseHz: float = 100,
                                     ):
        '''配置机载Lidar文件
        一、输入参数
        一、以将激光雷达z轴水平放置, 与航线方向相同(Airborne_scan中旋转Lidar)
        二、一圈多次tick,一个tick只有一个角度(原始角度)
        先根据角度分辨率计算每圈需要的tick次数, scanRateBaseHz是输入值,reportRateBaseHz等于每圈tick次数*scanRateBaseHz
        角度分辨率决定了每圈需要的tick次数,即每圈tick次数=FOV_angle/horizontal_angle_resolution
        三、发射器状态参数
        因为一个tick只有一个角度(原始角度),numberOfEmitters=1,azimuthDeg=elevationDeg=fireTimeNs=0
        '''

        start_angle = FOV_angle/2
        reportRateBaseHz = (FOV_angle/horizontal_angle_resolution)*scanRateBaseHz

        # 读写Lidar配置文件
        with open(Airborne_lidar_json_path, 'r') as f:
            data = json.load(f)
        # 修改数据
        data['profile']['validStartAzimuthDeg'] = 360-start_angle
        data['profile']['validEndAzimuthDeg'] = start_angle
        data['profile']['reportRateBaseHz'] = reportRateBaseHz
        data['profile']['scanRateBaseHz'] = scanRateBaseHz
        data['profile']['numberOfEmitters'] = 1
        data['profile']['emitterStates'][0]['azimuthDeg'] = [0 for i in range(1)]
        data['profile']['emitterStates'][0]['elevationDeg'] = [0 for i in range(1)]
        data['profile']['emitterStates'][0]['fireTimeNs'] = [0 for i in range(1)]
        # 将修改后的数据写回JSON文件
        with open(Airborne_lidar_json_path, 'w') as f:
            json.dump(data, f, indent=3)

    # 创建lidar
    def creat_Liar(self, Lidar_config: str = '',
                   position=(0, 0, 0)
                   ):
        '''创建Lidar'''
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/World/RTX_lidar/sensor",
            parent=None,
            config=Lidar_config,
            translation=position,
        )
        return sensor

    # 地基激光雷达扫描控制
    def Terrestrial_scan(self, carrier=None,
                         end_time: int = 30,
                         horizontal_start_angle: float = 0,
                         horizontal_end_angle: float = 360,
                         ):
        '''以载体旋转的方式控制地基激光雷达
        另外一种使用固态Lidar需配置多个反射器组状态(暂时放弃)。'''
        # 获取stage帧率
        time_codes = self.timeline.get_time_codes_per_seconds()
        # time_codes = timeline.set_time_codes_per_seconds()  设定舞台帧率

        # 旋转RTX_lidar
        OrientOp = carrier.AddRotateXYZOp()
        OrientOp.Set(time=0, value=(0, 0, horizontal_start_angle))
        Rotate_framerate = end_time*time_codes
        OrientOp.Set(time=Rotate_framerate, value=(0, 0, horizontal_end_angle))
        return end_time*time_codes

    # lidar载体路径规划与扫描控制
    def Airborne_scan(self, UAV_speed: int = 5,
                      scan_range: int = 100):
        '''移动控制总体思路
        一、未来的目标是要像大疆无人机app一样, 选定一个区域，然后自动规划路线这里涉及到将实际的三维场景先二维化(top视角),
        输出到第二个窗口选择观测范围，目前就直接手动输入起始点，航线间隔，自动规划。
        二、无人机控制参数包括高度、飞行速度
        '''
        # 旋转激光雷达，z轴水平放置
        sensor_prim = self.stage.GetPrimAtPath("/World/RTX_lidar/sensor")
        quat = self.euler2quat(0, 90, 0)   # 绕xyz旋转
        sensor_prim.GetAttribute('xformOp:orient').Set(quat)

        # 创建载体，与设定起始位置
        lidar_carrier_prim, xformable = self.creat_xform()
        lidar_carrier_prim.GetAttribute('xformOp:translate').Set((0, 0, 10))

        # 计算航线数量,每条航线起始位置
        interval = 5
        start_x, start_y, end_x, end_y, = 0, 0, 30, 40
        x_width = abs(start_x - end_x)
        y_width = abs(start_y - end_y)
        flight_number = math.floor(x_width/interval) + 1

        # 移动RTX_lidar载体
        time_codes = self.timeline.get_time_codes_per_seconds()
        # 单次扫描长度是20m
        single_line_time = (20/UAV_speed)*time_codes
        translateOp = xformable.AddTranslateOp()
        for i in range(flight_number):
            translateOp.Set(time=single_line_time*i+1, value=Gf.Vec3f(start_x, interval*i, 10))
            translateOp.Set(time=single_line_time*(i+1), value=Gf.Vec3f(end_x, interval*i, 10))

        # 计算扫描总时长
        return single_line_time*flight_number

    # 欧拉角转四元数
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

    # 点云可视化
    def Lidar_Visualisation(self, sensor):
        '''点云可视化'''
        #  Create and Attach a render product to the camera
        render_product_toVisualisation = rep.create.render_product(sensor.GetPath(), [1, 1])

        # Create a Replicator Writer that "writes" points into the scene for debug viewing
        writer_Visualisation = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        writer_Visualisation.attach(render_product_toVisualisation)

    # objectId转物体类型
    def Id2class(self):
        '''npy'''
        pass
