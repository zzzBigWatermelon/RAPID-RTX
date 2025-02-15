import os
import omni
import omni.ext
import asyncio
import omni.kit.commands
# 添加使用相对路径时使用到的Path函数
from pathlib import Path
# command命令的使用
import omni.kit.commands
# omni.usd命令的使用
import omni.usd
# pixar命令，Sdf是某种路径转换，UsdGeom是几何
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import pandas as pd
# 传感器模块
# from omni.isaac.range_sensor import _range_sensor
# from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
'''from .extension import on_clik_Simecircular, on_clik_Simecircular_Delete'''


# ---------------------------------------------code开发环境---------------------------------
'''这里规定Z轴负方向为正北方向,x轴的负方向为正西方向 所以z轴负方向为0度 作为旋转的起点 其他输入的参数不涉及设定旋转方向的依旧以z轴正向为起点 即默认的rotate起点
rotate属性的值顺时针旋转变小 逆时针旋转变大
cube的point属性(8组3个50)好像采用的是cm为单位 一个标准cube就是1m*1m*1m
物体translate属性的单位是cm 输入的参数空间距离单位都是cm 变换成以m为单位
默认相机初始的状态 镜头朝向z轴负方向 向上抬头rotateX变大为正值 向下变小为负值
'''
# ----------------------------------------------isaac开发环境-----------------------------------------------
'''默认单位时m 默认z轴向上 将默认的物体生成属性orient更改为了rotate'''
# 样本树木的加载路径
tree_01 = Path(__file__).parent.parent/'data'/'Model'/'Poplart_omniSurface.usd'  # omniSurface材质树木
# tree_01 = Path(__file__).parent.parent/'data'/'conifer1.usd'
# 机载激光雷达加载路径
lidarPath = Path(__file__).parent.parent/'data'/'Model'/'demo-fly.usd'
# 背包激光雷达加载路径
backpackPath = Path(__file__).parent.parent/'data'/'Model'/'demo-car.usd'
# CrazyFlie无人机模型路径
UAVModelAssetPath = Path(__file__).parent.parent/'data'/'Model'/'UAVModel_CrazyFlie.usd'
# UAVLidar模型路径
UAVLidarModelAssetPath = Path(__file__).parent.parent/'data'/'Model'/'UAVLidarModel.usd'
# 储存结果文件夹的路径
output_dir = Path(__file__).parent.parent/'result'
# 语义标签文件夹和语义标签文件
output_idToLabel_dir = Path(__file__).parent.parent/'result'/'idToLabel'
idToLabel_doc = Path(__file__).parent.parent/'result'/'idToLabel'/'semantic_segmentation_labels_0000.json'


class TreeGenerator:
    def __init__(self, width=0, length=0, TreeSpecies=0) -> None:
        '''self.width = width   # y方向
        self.length = length   # x方向
        self.thickness = thickness'''
        self.width = width
        self.length = length
        self.TreeSpecies = TreeSpecies
        # 是获得已经打开的舞台。,或者使我们能够访问已经加载的状态，在应用中。是获得已经打开的舞台。
        self._stage: Usd.Stage = omni.usd.get_context().get_stage()
        # 路径有效性检验，作用应该是此路径如果之前已经存在了，会再建立一个新路径，原路径后面加上序号，这一步不会生成文件夹
        self.geom_scope_path = Sdf.Path(omni.usd.get_stage_next_free_path(self._stage, '/World/Geometry', False))
        self.geom_looks_path = Sdf.Path(omni.usd.get_stage_next_free_path(self._stage, '/World/Looks', False))
        # 创建Geometry和Looks文件夹，command命令
        omni.kit.commands.execute(
            'CreatePrimWithDefaultXform',
            prim_type='Scope',
            prim_path=self.geom_scope_path,
            attributes={},
            select_new_prim=False)
        omni.kit.commands.execute(
            'CreatePrimWithDefaultXform',
            prim_type='Scope',
            prim_path=self.geom_looks_path,
            attributes={},
            select_new_prim=False)
        # ---------------------------------函数调用-------------------------------
        self.selection()
        self.create_prototype()
        self.create_TreeInstace()
        # ------------------------------添加动态天空背景--------------------------------
        '''omni.kit.commands.execute(
            'CreateDynamicSkyCommand',
            sky_url='https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Dynamic/CumulusLight.usd',
            sky_path='/Environment/sky')'''

    # -------------------------------返回当前选中的土地的路径,并计算他的偏移值，还有边界值--------------------------------------
    def selection(self):
        # 得到当前选中土地的路径和prim
        ctx = omni.usd.get_context()
        # returns a list of prim path strings
        selection_land = ctx.get_selection().get_selected_prim_paths()
        # selection_land返回一个列表，虽然不知道都有什么，但是用第一个列表元素能正确运行
        self.land_prim = self._stage.GetPrimAtPath(selection_land[0])
        # 获取偏移值,不是属性，而是属性中的值
        self.land_translate = self.land_prim.GetAttribute('xformOp:translate').Get()
        # 获取边界值，从下面获取每种树的边界改写而来,这里的边界值是物体未进行缩放前的值
        bbox_selection = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_selection.ComputeWorldBound(self.land_prim)
        bbox_range = bbox.GetRange()
        bbox_min = bbox_range.GetMin()
        bbox_max = bbox_range.GetMax()
        self.land_length = (bbox_max[0] - bbox_min[0])  # 得到地块的长度
        self.land_width = (bbox_max[1] - bbox_min[1])  # 得到地块的宽度
        # 上面得到的是物体原有的宽度和长度，如果有缩放存在，那么实际长宽高和真实的长宽高就存在差别，需要原有长宽高的值乘上缩放值。
        self.land_scale = self.land_prim.GetAttribute('xformOp:scale').Get()
        self.land_length = self.land_length * self.land_scale[0]
        self.land_width = self.land_width * self.land_scale[1]
        return self.land_translate

    # -----------------------------原型生成--------------------------------
    def create_prototype(self):
        '''# 创建一个不可见的原型Prototype
        self.prototypes: Usd.Prim = self._stage.OverridePrim(self.geom_scope_path.AppendPath('Prototype'))'''
        # 下面两个代码用于存放每个原型的长度、宽度和路径
        self.prototypes_lengths = []
        self.prototypes_widths = []
        self.prototypes_paths = []
        # 暂时不明白
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        # 下面的代码不太明白，应该是将原型书book添加到舞台上。再给书赋值给一个变量default_prim,
        # 暂时不明白
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        # 下面的代码不太明白，应该是将原型书添加到舞台上。再给书赋值给一个变量default_prim,
        # 单独测试时，下面代码并没有将原型文件加载到舞台上，
        """ tree_01_stage: Usd.Stage = Usd.Stage.Open(str(tree_01))
        tree_02_stage: Usd.Stage = Usd.Stage.Open(str(tree_02))
        tree_03_stage: Usd.Stage = Usd.Stage.Open(str(tree_03)) """
        # 三个原型的路径有效性检查,并添加到self.prototypes_paths
        self.tree_path = omni.usd.get_stage_next_free_path(
            self._stage,
            self.geom_scope_path.AppendPath('Tree'), False)
        # 通过command命令将tree加载到Prototype下，
        omni.kit.commands.execute(
            'CreatePayload',
            path_to=self.tree_path,
            asset_path=str(tree_01),
            usd_context=omni.usd.get_context())
        tree_prim = self._stage.GetPrimAtPath(self.tree_path)
        '''# -----------------------------------isaac中尺度标准不一样 需要修改变换-------------------------------
        American_Beech_path = '/World/Geometry/Prototype01/tree_01/American_Beech'
        tree_prim = self._stage.GetPrimAtPath(American_Beech_path)
        tree_scale = tree_prim.GetAttribute('xformOp:scale')
        tree_scale.Set(Gf.Vec3d(0.01, 0.01, 0.01))
        tree_rotate = tree_prim.GetAttribute('xformOp:rotateXYZ')
        tree_rotate.Set(Gf.Vec3d(0, 0, 0))'''
        # 下面是计算每种树的宽度和长度，依次来排列他们，并添加到 self.prototypes_widths
        # GetRange会得到两组数据，每组三个数值（应该是X,Y,Z的最大和最小值）。应该是物体两个极点的范围，
        bbox = bbox_cache.ComputeWorldBound(tree_prim)
        bbox_range = bbox.GetRange()
        bbox_min = bbox_range.GetMin()
        bbox_max = bbox_range.GetMax()
        self.prototypes_lengths.append((bbox_max[0] - bbox_min[0]))  # 得到每种树的长度
        self.prototypes_widths.append((bbox_max[1] - bbox_min[1]))  # 得到每种树的宽度
        # 应该是将书籍隐藏起来的功能
        '''self.prototypes01.SetSpecifier(Sdf.SpecifierOver)'''

    # --------------------------生成树的实例------------------------------------
    def create_TreeInstace(self):
        # ----------------------路径有效性检验----------------------------
        '''instancer01_path = Sdf.Path(omni.usd.get_stage_next_free_path(
            self._stage,
            self.geom_scope_path.AppendPath('Tree01Instacer'),
            False
        ))
        # 利用usd文档中的api创造实例 PointInstancer创造实例 Define应该是指定某一个prim充当实例
        instancer01 = UsdGeom.PointInstancer.Define(self._stage, instancer01_path)'''
        # 用于接受每个实例的空间位置和id
        '''positions = []
        proto_ids = []'''
        # -------------------------下面开始种树了----------------------------
        # 先种上一排,sum_width代表一排树的宽度总和来控制一排树木的数量,同时也能控制每一颗树的宽度
        # next_id下一次的id，next_width下一次的宽度
        sum_length = 0
        sum_width = 0
        '''下面代码是随机生成树种
        next_id = random.randint(0, len(self.prototypes_paths) - 1)'''
        # next_id生成的树种全部为指定的树种
        next_id = self.TreeSpecies
        # 选定指定树种的长和宽,此时长和宽用于设定起始位置
        next_width_01 = self.prototypes_widths[next_id]
        next_length_01 = self.prototypes_lengths[next_id]
        # 设定树木之间的间隙
        next_width = self.prototypes_widths[next_id] + self.width
        next_length = self.prototypes_lengths[next_id] + self.length
        # 通过api，给书的实例添加transla属性，设置空间位置,应该是相当于所有树实例的起点。
        # 这里的transla属性和下面的positions属性不是一个东西，应该是在transla作为起点的基础上，通过position设定相对于起点的空间位置
        # 设定起点，此时还没有把第一颗树种下去
        x_translate = self.land_translate[0] - self.land_length/2 + next_length_01/2
        y_translate = self.land_translate[1] - self.land_width/2 + next_width_01/2
        tree_prim = self._stage.GetPrimAtPath(self.tree_path)
        xform = UsdGeom.Xformable(tree_prim)
        xform.AddTranslateOp()
        tree_translate = tree_prim.GetAttribute('xformOp:translate')
        tree_translate.Set(Gf.Vec3d(x_translate,  y_translate, 0))
        '''xform = UsdGeom.Xformable(tree_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(x_translate, y_translate, 0))'''
        # 用以一个固定的原型树种填充满整个选择区域
        item = 1
        while sum_length + next_length < self.land_length:
            while sum_width + next_width < self.land_width:
                # -----------------在这里执行重复种植操作-------------------------------
                next_path = self.geom_scope_path.AppendPath('Tree' + str(item))
                item += 1
                omni.kit.commands.execute(
                    'CopyPrim',
                    path_from=self.tree_path,
                    path_to=next_path,
                    exclusive_select=False,
                    copy_to_introducing_layer=False)
                next_tree = self._stage.GetPrimAtPath(next_path)
                tree_translate = next_tree.GetAttribute('xformOp:translate')
                tree_translate.Set(Gf.Vec3d(x_translate + sum_length,  y_translate + sum_width, 0))
                # ------------------之前的思路重复思路----------------------------
                '''positions.append(Gf.Vec3d(sum_length, sum_width, 0))
                proto_ids.append(next_id)'''
                sum_width += next_width
            sum_length += next_length
            sum_width = 0
        omni.kit.commands.execute(
            'DeletePrims',
            paths=['/World/Geometry/Tree'],
            destructive=False)
        '''下面的代码是随机选择以一个原型树种填充到下一个位置，直到充满整个选择区域
        while sum_length + next_length < self.land_length:
            while sum_width + next_width < self.land_width:
                if next_id == 0:
                    positions01.append(Gf.Vec3d(sum_length, 0, sum_width))
                    proto01_ids.append(next_id)
                elif next_id == 1:
                    positions02.append(Gf.Vec3d(sum_length, 0, sum_width))
                    proto02_ids.append(next_id)
                elif next_id == 2:
                    positions03.append(Gf.Vec3d(sum_length, 0, sum_width))
                    proto03_ids.append(next_id)
                sum_width += self.prototypes_widths[next_id]
                next_id = random.randint(0, len(self.prototypes_paths) - 1)
                next_width = self.prototypes_widths[next_id]
            next_id = random.randint(0, len(self.prototypes_paths) - 1)
            sum_length += self.prototypes_lengths[next_id]
            next_length = self.prototypes_lengths[next_id]
            sum_width = 0'''
        # ------------------之前实例化的思路-------------------------
        '''# 添加一个positions属性 不知道干嘛 可以接收列表类型,应该是设置每一颗树的空间位置
        instancer01.CreatePositionsAttr().Set(positions)
        # 添加一个原型索引属性，
        instancer01.CreateProtoIndicesAttr().Set(proto_ids)
        # 这是将原型和实例链接，产生书的实例
        instancer01.CreatePrototypesRel().SetTargets(self.tree_path)'''


# 添加一个以py脚本驱动的xform作为林上林一体扫描无人机的驱动体
# lidar附加在xform中，xform作为驱动体，同时在xform中添加一个无人机模型(作用是好看)，Lidar模型
class Add_ScriptUAV:
    def __init__(self) -> None:
        self._stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.load()

    # 这个函数用来新建一个Xform驱动体，并且在其上加载脚本
    def load(self):
        # 路径有效性检验，建立一个保存加载资产的文件夹
        self.geom_scope_path = Sdf.Path(omni.usd.get_stage_next_free_path(self._stage, '/World/Scrip_UAV', False))
        omni.kit.commands.execute(
            'CreatePrimWithDefaultXform',
            prim_type='Scope',
            prim_path=self.geom_scope_path,
            attributes={},
            select_new_prim=False)
        # xform路径有效性检查,
        UAVxform_path = omni.usd.get_stage_next_free_path(
            self._stage,
            self.geom_scope_path.AppendPath('UAVxform'), False)
        # 新建一个xform，其在stage中的名字为Scrip_UAV
        UsdGeom.Xform.Define(self._stage, UAVxform_path)
        # 添加一个无人机模型
        UAVmodelStagePath = Sdf.Path(UAVxform_path).AppendPath('UAVmodel')
        # UAVmodelStagePrim = self._stage.GetPrimAtPath(UAVmodelStagePath)
        omni.kit.commands.execute(
            'CreateReference',
            path_to=UAVmodelStagePath,
            asset_path=str(UAVModelAssetPath),
            usd_context=omni.usd.get_context())
        # 添加一个lidar模型，lidar的载体为xform时，移动时lidar的位置会滞后
        UAVLidarModelStagePath = Sdf.Path(UAVxform_path).AppendPath('UAVLidarModel')
        omni.kit.commands.execute(
            'CreatePayload',
            path_to=UAVLidarModelStagePath,
            asset_path=str(UAVLidarModelAssetPath),
            usd_context=omni.usd.get_context())
        # 在lidar模型上添加激光雷达
        LidarPath = Sdf.Path(UAVxform_path).AppendPath('Lidar')
        result, Lidar = omni.kit.commands.execute(
                        "RangeSensorCreateLidar",  path=LidarPath, parent=None, min_range=0.0, max_range=50.0,
                        draw_points=False, draw_lines=True, horizontal_fov=120.0, vertical_fov=30.0,
                        horizontal_resolution=1, vertical_resolution=0.6, rotation_rate=2,
                        high_lod=True, yaw_offset=0.0, enable_semantics=True)
        # 设定机载lidar的xform属性
        Lidar_prim = self._stage.GetPrimAtPath(LidarPath)
        Lidar_prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(90, 90, 0))
        # 添加脚本文件
        ScriptFilePath = Path(__file__).parent.parent/'Forest2/UAVScript.py'
        omni.kit.commands.execute(
            'ApplyScriptingAPICommand',
            paths=[UAVxform_path])
        omni.kit.commands.execute(
            'ChangeProperty',
            prop_path=Sdf.Path(str(UAVxform_path)+'.omni:scripting:scripts'),
            value=Sdf.AssetPathArray(1, (Sdf.AssetPath(os.path.abspath(ScriptFilePath)))),
            prev=None,
            target_layer=None)


# 这个类的功能是加载无人机
class Load_LiDAR:
    def __init__(self) -> None:
        self._stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.load()

    # 这个函数用来加载无人机
    def load(self):
        # 路径有效性检验，建立一个保存加载资产的文件夹
        self.geom_scope_path = Sdf.Path(omni.usd.get_stage_next_free_path(self._stage, '/World/Assets', False))
        omni.kit.commands.execute(
            'CreatePrimWithDefaultXform',
            prim_type='Scope',
            prim_path=self.geom_scope_path,
            attributes={},
            select_new_prim=False)
        # lidarUAV路径有效性检查,
        self.lidar_path = omni.usd.get_stage_next_free_path(
            self._stage,
            self.geom_scope_path.AppendPath('LiDAR'), False)
        # 通过command命令将lidarUAV加载到舞台上
        omni.kit.commands.execute(
            'CreatePayload',
            path_to=self.lidar_path,
            asset_path=str(lidarPath),
            usd_context=omni.usd.get_context())
        # 添加:translate属性
        lidar_prim = self._stage.GetPrimAtPath(self.lidar_path)
        lidar_attr = lidar_prim.CreateAttribute('xformOp:translate', Sdf.ValueTypeNames.Double3)
        lidar_attr.Set((0, 0, 0))


# 这个类的功能是加载背光激光雷达
class Load_backpack:
    def __init__(self) -> None:
        self._stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.load()

    # 这个函数用来加载无人机
    def load(self):
        # 路径有效性检验，建立一个保存加载资产的文件夹
        self.geom_scope_path = Sdf.Path(omni.usd.get_stage_next_free_path(self._stage, '/World/Assets', False))
        omni.kit.commands.execute(
            'CreatePrimWithDefaultXform',
            prim_type='Scope',
            prim_path=self.geom_scope_path,
            attributes={},
            select_new_prim=False)
        # backpack路径有效性检查,
        self.backpack_path = omni.usd.get_stage_next_free_path(
            self._stage,
            self.geom_scope_path.AppendPath('backpack'), False)
        # 通过command命令将lidarUAV加载到舞台上
        omni.kit.commands.execute(
            'CreatePayload',
            path_to=self.backpack_path,
            asset_path=str(backpackPath),
            usd_context=omni.usd.get_context())
        # 添加:translate属性
        backpack_prim = self._stage.GetPrimAtPath(self.backpack_path)
        backpack_attr = backpack_prim.CreateAttribute('xformOp:translate', Sdf.ValueTypeNames.Double3)
        backpack_attr.Set((0, 0, 0))


# 这个应该使中心点的激光雷达，产生一张中心深度图
class lidar():
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()                       # Used to access Geometry
        self.lidar()

    def lidar(self):
        lidarPath = "/World/Lidar"
        timeline = omni.timeline.get_timeline_interface()
        lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        lidar_sensor_interface = acquire_lidar_sensor_interface()

        async def get_lidar_param():                                    # Function to retrieve data from the LIDAR
            await omni.kit.app.get_app().next_update_async()            # wait one frame for data
            timeline.pause()                                            # Pause the simulation to populate the LIDAR's depth buffers
            depth = lidarInterface.get_linear_depth_data("/World"+lidarPath)
            '''zenith = lidarInterface.get_zenith_data("/World"+lidarPath)
            azimuth = lidarInterface.get_azimuth_data("/World"+lidarPath)'''
            point = lidar_sensor_interface.get_point_cloud_data("/World" + lidarPath)
            depth_list = depth.tolist()
            # --------------------------------------先将深度信息转换成一个列表-------------------------------------------
            depth_list1 = []
            for item in depth_list:
                for item_1 in item:
                    depth_list1.append(item_1)
            number = 0
            point_list = point.tolist()
            with open(r'C:\Users\ZZZ\Desktop\point_static.txt', 'w') as src_file:
                for item in point_list:
                    for item_1 in item:
                        point_cloud = str(item_1).replace('[', '')
                        point_cloud1 = point_cloud.replace(']', '')
                        if depth_list1[number] != 35.0:
                            src_file.writelines(point_cloud1 + ',' + str(depth_list1[number]) + '\r\n')
                        number += 1
        timeline.play()                                                 # Start the Simulation
        asyncio.ensure_future(get_lidar_param())                        # Only ask for data after sweep is complete


class Delete:
    def __init__(self):
        # 删除清空操作，自动删除之前所创建的东西，command中获得
        omni.kit.commands.execute(
            'DeletePrims',
            paths=['/World/Geometry', '/World/Looks'],
            destructive=False)


# 读取LESS的RAMI场景文件，并快速构建场景
class Scene_Construction:
    def __init__(self):
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.main()

    # 引用模型和修改属性
    def Construction(self, translate_X, translate_Y, rotate, tree_name):
        # 计算下个空路径
        ACPL_path = Sdf.Path(omni.usd.get_stage_next_free_path(self.stage, '/World/{}'.format(tree_name), False))
        # 创建xform，接受references
        ref_prim: Usd.Prim = UsdGeom.Xform.Define(self.stage, ACPL_path).GetPrim()
        # 添加references
        references: Usd.References = ref_prim.GetReferences()
        references.AddReference(assetPath=r'E:\Data\RAMI\USD\{}\Collected_{}\{}.usd'.format(tree_name, tree_name, tree_name),
                                primPath=Sdf.Path.emptyPath)

        # 添加和修改属性
        xform = UsdGeom.Xformable(ref_prim)
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp()
        xform.AddTranslateOp()
        ref_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(translate_X, translate_Y, 0.0))
        ref_prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(0.0, 0.0, rotate))

    # 读取instances文件
    def read_xlsx(self, tree_name):
        raw_date = pd.read_excel(r'E:\Data\RAMI\USD\instances\{}.xlsx'.format(tree_name), header=0)
        data = raw_date.values  # 将pd数据转换为np数据
        for i in range(data.shape[0]):
            self.Construction(data[i, 0], data[i, 1], data[i, 3], tree_name)

    # 对每个树种进行操作
    def main(self):
        doc_path = r'E:\Data\RAMI\USD\instances'
        doc_name_list = os.listdir(doc_path)
        for name in doc_name_list:
            tree_name = str(name).rsplit('.', 1)[0]  # 提取树种名
            self.read_xlsx(tree_name)


# 读取LESS的RAMI场景文件，并快速构建场景
class wind_Scene_Construction:
    def __init__(self):
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.read_xlsx()

    # 引用模型和修改属性
    def Construction(self, translate_X, translate_Y, rotate):
        # 计算下个空路径
        next_path = Sdf.Path(omni.usd.get_stage_next_free_path(self.stage, '/World/tree', False))
        omni.kit.commands.execute('CopyPrim',
                                  path_from='/World/tree',
                                  path_to=next_path,
                                  exclusive_select=False,
                                  copy_to_introducing_layer=False)
        next_prim = self.stage.GetPrimAtPath(next_path)

        # 添加和修改属性
        next_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(translate_X, translate_Y, 0.0))
        next_prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3d(90, 0.0, rotate))

    # 读取instances文件
    def read_xlsx(self):
        raw_date = pd.read_excel(r'C:\Users\ZZZ\Desktop\paper\wind_scenes\instances.xlsx', header=0)
        data = raw_date.values  # 将pd数据转换为np数据
        for i in range(data.shape[0]):
            self.Construction(data[i, 1], data[i, 2], data[i, 4])
