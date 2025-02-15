from pxr import Usd, UsdGeom, Gf, Sdf
import omni
import omni.usd
import omni.kit.app
import time
import asyncio
import omni.replicator.core as rep
import math
from pathlib import Path
from .WorkWriter import WorkWriter

'''rep使用的是code2022.3.3版本,此版本的replicator为1.7.7,最新的repAPI中有些接口不能使用, 如rep.create.xform'''
'''更新:rep使用的是composer2023.2.5版本,此版本的replicator为1.10.10'''


# 无人机资产路径
UAV_model_assets = r'C:\Users\ZZZ\Desktop\UAVModel.usd'


# 生成光源，控制光源初始的状态
def lightPositions():

    # 太阳光源变化模拟，要创建两个xform
    # 第一个xform，控制光源保持在太阳视平面，同时改变角度可以直接控制太阳的最高天顶角（方位角为0时）
    # 天顶角0到60对应light_parent_parent_prim旋转属性的-90到-30
    light_parent_parent_prim = rep.create.xform(position=(0.0, 0.0, 0.0),
                                                rotation=(0.0, -70.0, 0.0),
                                                name='direct_Light_parent_parent')
    # 第二个xform，以ratation的z轴数值控制光源在太阳视平面的旋转
    light_parent_prim = rep.create.xform(position=(0.0, 0.0, 0.0),
                                         rotation=(0.0, 0.0, 0.0),
                                         name='light_parent',
                                         parent='/Replicator/direct_Light_parent_parent')
    # 直射光源本身控制平移距离属性
    direct_light = rep.create.light(light_type="disk",
                                    position=(10000000*(2**0.5), 0.0, 0.0),
                                    scale=360,  # disk光源默认半径为0.5（原场景光源半径为60）
                                    look_at=(0, 0, 0),
                                    intensity=4099982592.0,
                                    parent='/Replicator/direct_Light_parent_parent/light_parent')
    # ---------------------------------散射光源--------------------------------------
    # 做直射与散射比例对无人机阴影在热点处的影响，
    '''scattering_light = rep.create.light(light_type="dome",
                                        intensity=0,
                                        name='scattering_Light')'''
    # light_parent_prim控制光源在视平面上的运动，
    return light_parent_prim, light_parent_parent_prim, direct_light


# 控制光源运动，三种模式（固定，多SZA，太阳视运动）
def Light_control(num):
    # 光源参数，以飞行时间为输入，控制太阳运动的角度，再以rep.distribution均匀分布角度。
    light_speed = 360/24  # 光源速度设定，太阳360°/24h
    fly_time = 0.25  # 单位为小时
    # 计算每个采样点对应的太阳变化角度
    angle_change = light_speed * fly_time

    # 固定光源,属性去lightPositions()方法中修改
    fixed_light = True
    if fixed_light:
        light_parent_prim, light_parent_parent_prim, direct_light = lightPositions()

    # 直接改变太阳的天顶角（非太阳视运动）,天顶角0到60对应light_parent_parent_prim(第一个xform)旋转属性的-90到-30
    need_multiSZA_light = False
    if need_multiSZA_light:
        light_parent_prim, light_parent_parent_prim, direct_light = lightPositions()
        zenith_change_light_positions = [(0.0, -i*2.0, 0.0) for i in range(15, 46)]

        def zenith_change_light():
            with light_parent_parent_prim:
                rep.modify.attribute(name='xformOp:rotateXYZ',
                                     value=rep.distribution.sequence(zenith_change_light_positions),
                                     attribute_type='float')
            return light_parent_parent_prim.node
        rep.randomizer.register(zenith_change_light)

    # 太阳视运动(控制light_parent_parent_prim（第二个xform）z轴属性，从x轴正向（朝屏幕外）逆时针旋转）
    need_move_light = False
    if need_move_light:
        # 生成初始光源
        light_parent_prim, light_parent_parent_prim, direct_light = lightPositions()
        # 当太阳视平面的最高天顶角为30°时（正午12点）方位角为0，视平面偏移0,太阳运动到天顶角45°时，方位角为55，视平面偏移35.5
        # 太阳运动到天顶角60°时，方位角为70.5，视平面偏移54.5
        angle_change_step = float(round((angle_change / num), 2))
        light_positions = [(0.0, 0.0, angle_change_step*i+0) for i in range(num+1)]

        def move_light():
            with light_parent_prim:
                rep.modify.attribute(name='xformOp:rotateXYZ',
                                     value=rep.distribution.sequence(light_positions),
                                     attribute_type='float')
            return light_parent_prim.node
        rep.randomizer.register(move_light)
    return need_multiSZA_light, need_move_light


# 生成相机和控制飞行方式
# 三种模式，固定相机、竖直飞行、水平圆盘飞行
def cameraPositions():
    height = 2000  # 相机距离中心点的距离h>2000（焦距>2500）米以上时，可以近似正交相机

    # 三种模式，固定相机、竖直飞行、水平圆盘飞行
    fixed_camera = False
    multiVZA_vertical = True   # 选择竖直观测（zenith天顶角变化）
    multiVZA_horizontal = False  # 水平圆盘观测时从x轴正向逆时针旋转（azimuth方位角变化）

    # 记录相机空间位置
    camera_positions = []
    # 记录无人机模型空间位置（有影子影响的情况下）
    UAV_model_positions = []
    num = 0

    # 生成相机模块
    camera = rep.create.camera(position=(0, 0, 0),
                               look_at=(0, 0, 0),
                               focal_length=2500,
                               name='camera_BRF')

    # 第一种模式，固定位置观测
    if fixed_camera:
        camera_positions.append([0, 0, 80])
        UAV_model_positions.append([0, 0, 80+0.2])
        num = 1

    # 第二种模式——竖直飞行
    if multiVZA_vertical:
        rotationStep = 1  # 旋转步长设定为5 (竖直观测)
        azimuth = math.radians(0)  # 方位角默认为0，即x轴平面运动
        # 产生一系列的角度值（天顶角），从其中条选择半圆角度
        angleList = [a*rotationStep for a in range(30, 151)]

        # 竖直半圆环式观测位置计算（天顶角变动）
        for i in angleList:
            # python使用弧度制，角度需要先转成弧度
            if 151 > i > 29:
                Zenith = math.radians(i)
                x = height*math.cos(Zenith)
                y = 0
                z = height*math.sin(Zenith)
                camera_positions.append((x, y, z))
                # 无人机模型的位置（有影子影响的情况下）
                UAV_model_positions.append((x, y, z+0.2))
        num = len(camera_positions)  # 一次总采样包含的采样点个数

    # 第三种模式——水平圆盘飞行
    if multiVZA_horizontal:
        rotationStep = 5  # 旋转步长设定为10 (水平观测)
        Zenith = math.radians(42)   # 无人机的高度角
        # 产生一系列的角度值（方位角），整圆角度（0-360），以x轴正向为0°
        angleList = [a*rotationStep for a in range(0, 36)]
        for i in angleList:
            # 水平圆环式观测位置计算（方位角变动）
            # python使用弧度制，角度需要先转成弧度
            azimuth = math.radians(i)
            # 计算相机空间位置
            x = height*math.cos(Zenith)*math.cos(azimuth)
            y = height*math.cos(Zenith)*math.sin(azimuth)
            z = height*math.sin(Zenith)
            camera_positions.append((x, y, z))
            # 无人机模型的位置（有影子影响的情况下）
            UAV_model_positions.append((x, y, z+0.2))
        # 在最后一个飞行位置增加一次最开始位置的观测
        UAV_model_positions.append(UAV_model_positions[0])
        camera_positions.append(camera_positions[0])
        num = len(camera_positions)  # 一次总采样包含的采样点个数

    return camera_positions, num, UAV_model_positions, camera


class BRF_viewCamera:
    def __init__(self) -> None:
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()

        # 函数直接调用（调试）
        # self.raplicatorBRF()  # 竖直半圆和水平圆盘飞行
        # self.cameraPositions_wind()  # 风立场相机

        # 异步编程处理（I/O优化,或增加raplicator的后续操作）
        async def go():
            self.raplicatorBRF()
            await rep.orchestrator.run_async()
            start = time.time()
            await rep.orchestrator.run_until_complete_async()

            # 计算时间
            end_time = time.time()
            execution_time = end_time - start
            print('----------------------------------------------------------------------------------')
            print(execution_time)
            print('----------------------------------------------------------------------------------')
            # 删除本次raplicator模块的
            omni.kit.commands.execute('DeletePrims',
                                      paths=['/Replicator'],
                                      destructive=False)
        asyncio.ensure_future(go())

    def raplicatorBRF(self):

        # 设定观测林子BRF的相机
        camera_positions, num, UAV_model_positions, camera = cameraPositions()
        # num = 5

        # 携带相机的无人机模型（模拟有阴影情况下的观测）
        need_uav_model = False
        if need_uav_model:
            def move_UAV():
                UAV_model = rep.create.from_usd(UAV_model_assets, semantics=[('class', 'UAV')])
                with UAV_model:
                    rep.modify.pose(position=rep.distribution.sequence(UAV_model_positions))
                return UAV_model.node
            rep.randomizer.register(move_UAV)

        # 光源控制
        need_multiSZA_light, need_move_light = Light_control(num)

        # Set the renderer to Path Traced,采用4：3的像素比例
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
        render_product = rep.create.render_product(camera, resolution=(500, 500))

        # 触发器语句
        with rep.trigger.on_frame(num_frames=num, rt_subframes=5):
            # 控制无人机的运动
            if need_uav_model:
                rep.randomizer.move_UAV()
            # 控制多SZA的光源
            if need_multiSZA_light:
                rep.randomizer.zenith_change_light()
            # 太阳视运动
            if need_move_light:
                rep.randomizer.move_light()

            with camera:
                rep.modify.pose(position=rep.distribution.sequence(camera_positions), look_at=(0, 0, 0))
            '''时间轴上采样, 模拟风对林地BRF观测影响时使用
            rep.modify.time(rep.distribution.choice(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                                                    with_replacements=False,
                                                    name='value'))'''

        # Initialize and attach writer
        writer = rep.WriterRegistry.get("WorkWriter")
        writer.initialize(
            output_dir=Path(__file__).parent.parent/'result'/'BRF'/'view',
            rgb=True)
        writer.attach([render_product])

    # 风立场下的观测模拟,相机位置不变，持续观测一个地点
    def cameraPositions_wind(self):
        # 观测林子BRF热点的相机
        camera = rep.create.camera(
            position=(20, 20, 5500),
            look_at=(20, 20, 0),
            focal_length=2500,
            name='camera_BRFhot')
        # Set the renderer to Path Traced
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
        render_product = rep.create.render_product(camera, (500, 500))
        with rep.trigger.on_frame(max_execs=1):
            # rep.modify.time(5)
            '''rep.modify.time(rep.distribution.choice(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  21],
                                                    with_replacements=False,
                                                    name='value'))'''
        # Initialize and attach writer
        '''writer = MyWriter()
        writer.attach(render_product)'''
        writer = rep.WriterRegistry.get("WorkWriter")
        writer.initialize(
            output_dir=Path(__file__).parent.parent/'result'/'BRF'/'view',
            rgb=True,
            bounding_box_2d_tight=True,
            semantic_types=['class'])
        # distribution_name=True
        writer.attach([render_product])
        rep.orchestrator.run()


# 场景中必须要有一个'/World/target'的靶标才能正常运行
class viewGrayScale:
    def __init__(self) -> None:
        self.height = 2000   # 相机高度
        # 灰阶靶标正上方的一次观测方位
        self._stage: Usd.Stage = omni.usd.get_context().get_stage()
        target = self._stage.GetPrimAtPath('/World/target')
        self.camera2AttrX = target.GetAttribute('xformOp:translate').Get()[0]
        self.camera2AttrY = target.GetAttribute('xformOp:translate').Get()[1]
        self.grayScale()

    # 观测一次灰阶靶标
    def grayScale(self):
        # 相机设定
        camera_position = [(self.camera2AttrX, self.camera2AttrY, self.height)]
        camera = rep.create.camera(
            position=(0, 0, 2000),
            look_at=(0, 0, 0),
            focal_length=2500,
            name='camera_grayScale')

        # 渲染设定
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
        render_product = rep.create.render_product(camera, (500, 500))
        num = 1

        # 光源设定
        need_multiSZA_light, need_move_light = Light_control(num)

        # -----------------------------有无人机阴影的热点观测，直接改变SZA太阳天顶角（辐射定标）--------------
        # 一般情况下不需要在定标中多次改变天顶角
        zenith_change_light_bool = False
        if zenith_change_light_bool:
            # 光源设定
            light_parent_prim, light_parent_parent_prim, direct_light = lightPositions()
            # 直接改变太阳的天顶角（非太阳视运动）,天顶角0到60对应Olight_parent_parent_prim旋转属性的-90到-30
            zenith_change_light_positions = [(0.0, -i*2.0, 0.0) for i in range(15, 46)]
            # 有无人机阴影的热点观测，SZA和VZA同步变化

            def zenith_change_light():
                with light_parent_parent_prim:
                    rep.modify.attribute(name='xformOp:rotateXYZ',
                                         value=rep.distribution.sequence(zenith_change_light_positions),
                                         attribute_type='float')
                return light_parent_parent_prim.node
            rep.randomizer.register(zenith_change_light)
            num = 31

        # 触发器，数据输出
        with rep.trigger.on_frame(num_frames=num, rt_subframes=5):
            '''太阳天顶角多次变化（非太阳视运动）'''
            # rep.randomizer.zenith_change_light()
            with camera:
                rep.modify.pose(position=rep.distribution.sequence(camera_position),
                                look_at=(self.camera2AttrX, self.camera2AttrY, 0))
        writer = rep.WriterRegistry.get("WorkWriter")
        writer.initialize(
            output_dir=Path(__file__).parent.parent/'result'/'BRF'/'view'/'Gray-Scale Targets',
            rgb=True,
            bounding_box_2d_tight=True,
            semantic_types=['target'])
        writer.attach([render_product])
        rep.orchestrator.run()
