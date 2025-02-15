from omni.kit.scripting import BehaviorScript
from pxr import Usd, Gf, Sdf, UsdGeom, Semantics
from omni.isaac.range_sensor import _range_sensor
import omni.usd
import omni
import omni.timeline
import omni.kit.app
import omni.kit.commands
import omni.timeline
from pathlib import Path
import os
import json


# 林上林下扫描无人机脚本
class NewScript(BehaviorScript):
    def on_init(self):
        print(f"{__class__.__name__}.on_init()->{self.prim_path}")
        # 通过xform设定移动
        xform = UsdGeom.Xformable(self.prim)   # self.prim应该是被脚本所附加的
        xform.ClearXformOpOrder()
        offset = xform.AddTranslateOp(opSuffix='offset')
        # 获取一些载体xform和lidar基本属性
        self.LidarPath = Sdf.Path(self.prim_path).AppendPath('Lidar')
        self.StrLidarPath = str(self.LidarPath)
        self.LidarPrim = self.stage.GetPrimAtPath(self.LidarPath)
        # 测试机翼旋转
        '''Spin1Prim = self.stage.GetPrimAtPath(Sdf.Path(self.prim_path).AppendPath('UAVmodel/m1_prop'))
        Spin2Prim = self.stage.GetPrimAtPath(Sdf.Path(self.prim_path).AppendPath('UAVmodel/m2_prop'))
        Spin3Prim = self.stage.GetPrimAtPath(Sdf.Path(self.prim_path).AppendPath('UAVmodel/m3_prop'))
        Spin4Prim = self.stage.GetPrimAtPath(Sdf.Path(self.prim_path).AppendPath('UAVmodel/m4_prop'))
        Spin1Xform = UsdGeom.Xformable(Spin1Prim)
        Spin2Xform = UsdGeom.Xformable(Spin2Prim)
        Spin3Xform = UsdGeom.Xformable(Spin3Prim)
        Spin4Xform = UsdGeom.Xformable(Spin4Prim)
        Spin1Xform.ClearXformOpOrder()
        Spin2Xform.ClearXformOpOrder()
        Spin3Xform.ClearXformOpOrder()
        Spin4Xform.ClearXformOpOrder()
        offset1 = Spin1Xform.AddTranslateOp(opSuffix='offset')
        offset2 = Spin2Xform.AddTranslateOp(opSuffix='offset')
        offset3 = Spin3Xform.AddTranslateOp(opSuffix='offset')
        offset4 = Spin4Xform.AddTranslateOp(opSuffix='offset')
        Spin1 = Spin1Xform.AddRotateZOp(opSuffix='spin')
        Spin2 = Spin2Xform.AddRotateZOp(opSuffix='spin')
        Spin3 = Spin3Xform.AddRotateZOp(opSuffix='spin')
        Spin4 = Spin4Xform.AddRotateZOp(opSuffix='spin')
        Spin1.Set(time=1, value=0)
        Spin1.Set(time=4000, value=51440)
        offset1.Set(time=1, value=(0.03, -0.03, 0.02))
        offset1.Set(time=4000, value=(0.03, -0.03, 0.02))
        Spin2.Set(time=1, value=0)
        Spin2.Set(time=4000, value=51440)
        offset2.Set(time=1, value=(0.03, 0.03, 0.02))
        offset2.Set(time=4000, value=(0.03, 0.03, 0.02))
        Spin3.Set(time=1, value=0)
        Spin3.Set(time=4000, value=51440)
        offset3.Set(time=1, value=(-0.03, -0.03, 0.02))
        offset3.Set(time=4000, value=(-0.03, -0.03, 0.02))
        Spin4.Set(time=1, value=0)
        Spin4.Set(time=4000, value=51440)
        offset4.Set(time=1, value=(-0.03, 0.03, 0.02))
        offset4.Set(time=4000, value=(-0.03, 0.03, 0.02))'''
        # 怀来场景
        offset.Set(time=500, value=(87, 92, 25))
        offset.Set(time=4100, value=(51, 92, 25))
        offset.Set(time=4900, value=(51, 84, 25))
        offset.Set(time=8500, value=(87, 84, 25))
        offset.Set(time=9200, value=(87, 77, 25))
        offset.Set(time=12800, value=(51, 77, 25))   # 机载完成
        offset.Set(time=13800, value=(51, 77, 2))   # 下降
        offset.Set(time=17400, value=(87, 77, 2))
        offset.Set(time=18100, value=(87, 84, 2))
        offset.Set(time=21700, value=(51, 84, 2))
        offset.Set(time=22500, value=(51, 92, 2))
        offset.Set(time=26100, value=(87, 92, 2))
        '''# 测试移动----测试场景
        offset.Set(time=500, value=(20, 20, 25))
        offset.Set(time=4500, value=(-20, 20, 25))
        offset.Set(time=5500, value=(-20, 10, 25))
        offset.Set(time=9500, value=(20, 10, 25))
        offset.Set(time=10500, value=(20, 0, 25))
        offset.Set(time=14500, value=(-20, 0, 25))
        offset.Set(time=15500, value=(-20, -10, 25))
        offset.Set(time=19500, value=(20, -10, 25))
        offset.Set(time=20500, value=(20, -20, 25))
        offset.Set(time=24500, value=(-20, -20, 25))  # 机载扫描完成
        offset.Set(time=26333, value=(-19.71, -16.53, 1))
        offset.Set(time=27197, value=(-11.83, -20.09, 1))
        offset.Set(time=28166, value=(-5.41, -12.83, 1))
        offset.Set(time=28712, value=(0.05, -12.83, 1))
        offset.Set(time=29311, value=(5.69, -14.84, 1))
        offset.Set(time=30637, value=(18.89, -13.62, 1))  # path5
        offset.Set(time=31291, value=(16.99, -7.36, 1))
        offset.Set(time=32067, value=(14.81, 0.09, 1))
        offset.Set(time=32787, value=(8.55, -3.46, 1))
        offset.Set(time=33916, value=(-2.33, -6.48, 1))
        offset.Set(time=35278, value=(-15.63, -3.56, 1))
        offset.Set(time=35871, value=(-19.2, 1.18, 1))   # path11
        offset.Set(time=37240, value=(-23.16, 14.28, 1))
        offset.Set(time=37987, value=(-18.85, 19.24, 1))
        offset.Set(time=38515, value=(-14.77, 23.88, 1))
        offset.Set(time=39355, value=(-6.37, 23.88, 1))
        offset.Set(time=40137, value=(1.2, 25.87, 1))
        offset.Set(time=40743, value=(5.99, 22.16, 1))
        offset.Set(time=41818, value=(16.63, 20.6, 1))'''
        # 激光雷达事件流
        self._editor_event_subscription_aerialLidar = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_editor_step_aerialLidar)
        )
        # 读取语义分割文件
        idToLabel_file = Path(__file__).parent.parent/'result'/'pointcloud'/'idToLabel'/'semantic_segmentation_labels_0000.json'
        if os.path.exists(idToLabel_file):
            with open(idToLabel_file, 'r') as f:
                self.idToLabel = json.load(f)   # 返回字典对象
        else:
            self.idToLabel = 0

    def on_destroy(self):
        print(f"{__class__.__name__}.on_destroy()->{self.prim_path}")
        self._timeline_sub = None

    def on_play(self):
        # 对timeline的设定
        # self.stage.SetFramesPerSecond(50)  # 设定实际timeline的每秒帧数，但是timeline显示的帧数没改变
        self.FramesPerSecond = self.timeline.get_time_codes_per_seconds()
        print(f"{__class__.__name__}.on_play()->{self.prim_path}")

    def on_pause(self):
        print(f"{__class__.__name__}.on_pause()->{self.prim_path}")

    def on_stop(self):
        print(f"{__class__.__name__}.on_stop()->{self.prim_path}")
        self._editor_event_subscription_aerialLidar = None

    def on_update(self, current_time: float, delta_time: float):
        CurrentTimeSecond = self.timeline.get_current_time()  # 获取当前时间，单位是秒
        CurrentTimeFrames = round(CurrentTimeSecond*self.FramesPerSecond)  # 乘以当前每秒帧数后简单四舍五入获得当前帧
        XformAttr_Offset = self.prim.GetAttribute('xformOp:translate:offset').Get(CurrentTimeFrames)
        # 改变lidar属性，使其在较低高度时从机载扫描模式改变成为背包扫描模式
        if XformAttr_Offset[2] <= 2:
            self.LidarPrim.GetAttribute('horizontalFov').Set(360)
            self.LidarPrim.GetAttribute('horizontalResolution').Set(2)
            self.LidarPrim.GetAttribute('rotationRate').Set(10)
            self.LidarPrim.GetAttribute('verticalFov').Set(30)
            self.LidarPrim.GetAttribute('verticalResolution').Set(2)
            self.LidarPrim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(0, 0, 0))
        # print(f"{__class__.__name__}.on_update(current_time={current_time}, delta_time={delta_time})->{self.prim_path}")

    # -----------------------------------激光雷达数据获取功能----------------------------------
    # 空中激光雷达数据逻辑检测函数,循环订阅函数
    def _on_editor_step_aerialLidar(self, step):
        if self.timeline.is_playing():
            self._get_info_function_aerialLidar()

    # 空中激光雷达核心方法，用于获取、处理和生成数据
    def _get_info_function_aerialLidar(self):
        # 定义数据存储位置和舞台上激光雷达及其主体位置
        data_path = Path(__file__).parent.parent/'result'/'pointcloud'/'point_aerialLidar.txt'
        # 获取点云信息、深度信息、语义id分割标签信息（反射率信息，激光雷达所获取的反射率信息只有两个值，一个最大值255，一个最小值0，没有什么用处）
        lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        depth = lidarInterface.get_linear_depth_data(self.StrLidarPath)
        point = lidarInterface.get_point_cloud_data(self.StrLidarPath)
        semantics = lidarInterface.get_semantic_data(self.StrLidarPath)
        azimuth = lidarInterface.get_azimuth_data(self.StrLidarPath)
        # 转换数据形式，numpy.ndarray转换到列表
        point_list = point.tolist()
        depth_list = depth.tolist()
        semantics_list = semantics.tolist()
        azimuth_list = azimuth.tolist()
        # 点云信息转换，get_point_cloud_data得到的时相对于传感器原点的点云xyz，需要加上xform的位置
        point_cloud = []
        FramesPerSecond = self.timeline.get_time_codes_per_seconds()
        CurrentTimeSecond = self.timeline.get_current_time()  # 获取当前时间，单位是秒
        CurrentTimeFrames = round(CurrentTimeSecond*FramesPerSecond)  # 乘以当前每秒帧数后简单四舍五入获得当前帧
        body_translate = self.prim.GetAttribute('xformOp:translate:offset').Get(CurrentTimeFrames)
        # 这里分开机载和背包扫描数据的空间变换
        if body_translate[2] <= 2:
            for point_1 in point_list:
                for point_2 in point_1:
                    point_cloud.append([body_translate[0] + point_2[0],
                                        body_translate[1] + point_2[1],
                                        body_translate[2] + point_2[2]])
        # 由于机载机关雷达的雷达扫描方式是z字扫描，所以这里的雷达实际上已经在x轴旋转正90度，在y轴旋转正90度
        # 经过旋转后，激光雷达的自身坐标系相对于无人机（整个空间）发生了变化，无人机x轴对应雷达y轴，无人机y轴对应雷达负z轴，无人机z轴对应雷达负x轴
        # 雷达得到的原始点云和无人机移动距离相加时，要保持雷达每个轴的数据加在正确的无人机每个轴上
        else:
            for point_1 in point_list:
                for point_2 in point_1:
                    point_cloud.append([body_translate[0] + point_2[1],
                                        body_translate[1] - point_2[2],
                                        body_translate[2] - point_2[0]])
        # 先将深度信息转换成一个列表，再用number控制添加到每行文本的深度值
        depth_list1 = []
        for item in depth_list:
            for item_1 in item:
                depth_list1.append(item_1)
        # 转换语义id信息
        semantics_list1 = []
        for semantics_list_1 in semantics_list:
            for semantics_list_2 in semantics_list_1:
                semantics_list1.append(semantics_list_2)
        # 转换语义id信息
        azimuth_list1 = []
        for azimuth_list_1 in azimuth_list:
            azimuth_list1.append(azimuth_list_1)
        # 输出到文本中,因为point、depth和label数量相等，一一对应。可以用depth_list1[number]来剔除lidar边界值点(固定最大深度为35)
        number = 0
        with open(data_path, 'a') as src_file:
            for point_cloud1 in point_cloud:
                # 去掉[]，方便后续操作,达到特定输出格式
                point_cloud1 = str(point_cloud1).replace('[', '')
                point_cloud1 = point_cloud1.replace(']', '')
                if depth_list1[number] != 50.0 and depth_list1[number] > 2.0:
                    # 在这里区分一下有id标签文件（即点击了Semantic Segmentation按钮）和没有的情况
                    if self.idToLabel != 0:
                        # 这里保留小数点后6位,format函数只能用于数字类型变量,point_cloud1虽然包含3个轴的数据，但是一个字符串，不能对其中每个轴的数据进行操作，
                        # split先劈分point_cloud1，之后再操作每个轴的数据
                        point_split = point_cloud1.split(sep=',')
                        output = ('{:.6f}'.format(float(point_split[0])),   # x轴
                                  '{:.6f}'.format(float(point_split[1])),   # y轴
                                  '{:.6f}'.format(float(point_split[2])),   # z轴
                                  '{:.6f}'.format(float(depth_list1[number])),
                                  str(semantics_list1[number]), '\r\n')
                                # self.idToLabel[str(semantics_list1[number])]['class'],
                    else:
                        point_split = point_cloud1.split(sep=',')
                        output = ('{:.6f}'.format(float(point_split[0])),
                                  '{:.6f}'.format(float(point_split[1])),
                                  '{:.6f}'.format(float(point_split[2])),
                                  '{:.6f}'.format(float(depth_list1[number])),
                                  '''str(semantics_list1[number]), ''' '\r\n')
                    src_file.writelines(','.join(output))
                number += 1
