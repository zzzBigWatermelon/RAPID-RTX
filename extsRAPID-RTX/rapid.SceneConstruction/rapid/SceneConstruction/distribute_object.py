import numpy as np
import os
import re
import carb
import asyncio
from typing import Union, List
import omni.kit.raycast.query
import omni.usd
import omni.kit.commands
import pandas as pd
from pathlib import Path
from pxr import Usd, UsdGeom, Gf, Sdf, Vt
import random


class TerrainHeightSampler:
    def __init__(self, prim_path, dist_type, num, extent, mode="PointInstancer"):
        """
        Args:
            asset_path: 资产路径
            dist_type: 分布类型 ("Random" / "Uniform")
            num: 数量
            extent: 范围 [width, height]
            mode: 实例化模式 ("PointInstancer" 或 "PrimInstancing")
        """
        self.prim_path = prim_path
        self.dist_type = dist_type
        self.num = int(num)
        self.extent = extent
        self.mode = mode  # 保存模式

        self.completed_count = 0
        self.results_3d = None

    async def run(self):
        """外部入口"""
        # 1. 生成 XY 坐标
        random_xy = self._generate_xy()
        # 2. 射线采样地形获取高度和法线，返回XYZ位置和法线
        points_and_normals = await self._get_heights(random_xy)

        # 3. 根据模式选择分发方式
        if self.mode == "PointInstancer":
            await DistributionUtils.place_with_pointinstancer(self.prim_path, points_and_normals)
        elif self.mode == "PrimInstancing":
            await DistributionUtils.place_with_prim_instancing(self.prim_path, points_and_normals)

        print(f"[Sampler] {self.mode} 分发完成：{self.num} 个物体。")

    def _generate_xy(self):
        half_x, half_y = self.extent[0] / 2.0, self.extent[1] / 2.0
        if self.dist_type == "Random":
            return np.random.uniform(low=[-half_x, -half_y], high=[half_x, half_y], size=(self.num, 2))
        else:
            side = int(np.sqrt(self.num))
            x = np.linspace(-half_x, half_x, side)
            y = np.linspace(-half_y, half_y, side)
            xv, yv = np.meshgrid(x, y)
            return np.vstack([xv.ravel(), yv.ravel()]).T[:self.num]

    async def _get_heights(self, x_y_list):
        num_rays = len(x_y_list)
        self.results_3d = np.zeros((num_rays, 6), dtype=np.float32)
        self.completed_count = 0
        raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()

        for i in range(num_rays):
            px, py = x_y_list[i]
            ray = omni.kit.raycast.query.Ray((float(px), float(py), 10000.0), (0.0, 0.0, -1.0))
            raycast_interface.submit_raycast_query(ray, lambda r, res, idx=i, x=px, y=py: self._on_hit_callback(r, res, idx, x, y))

        while self.completed_count < num_rays:
            await asyncio.sleep(0.01)
        return self.results_3d

    def _on_hit_callback(self, ray, result, idx, x, y):
        if result.valid:
            self.results_3d[idx] = [x, y, result.hit_position[2], result.normal[0], result.normal[1], result.normal[2]]
        else:
            self.results_3d[idx] = [x, y, 0.0, 0.0, 0.0, 1.0]
        self.completed_count += 1


class ImportDataDistributor:
    def __init__(self, folder_path: str, data_file_path: str, mode="PointInstancer"):
        """
        Args:
            data_file_path: 位置数据文件路径
            mode: 实例化模式 ("PointInstancer" 或 "PrimInstancing")
        """
        self.folder_path = folder_path
        self.data_file_path = data_file_path
        self.mode = mode  # 保存模式

    async def run(self):
        """程序运行主要逻辑入口
        """
        # 读取csv文件数据
        object_paths, proto_indices, data_list = self.read_csv_file()
        # 根据模式调用不同的分发函数
        if self.mode == "PointInstancer":
            await DistributionUtils.place_with_pointinstancer(object_paths, data_list, proto_indices)
        else:
            await DistributionUtils.place_with_prim_instancing(object_paths, data_list, proto_indices)

    def read_csv_file(self):
        '''
        '''
        # 读取数据
        df = pd.read_csv(self.data_file_path)
        df.columns = df.columns.str.strip()

        # 取XYZ数据和文件名数据
        x = df.iloc[:, 0].astype(float).values
        y = df.iloc[:, 1].astype(float).values
        z = df.iloc[:, 2].astype(float).values
        names = df.iloc[:, 3].astype(str).str.strip().values

        # 提取唯一的名称列表，并保持出现顺序 (作为object_path)
        unique_names = []
        for n in names:
            if n not in unique_names:
                unique_names.append(n)
        # 构建完整的文件路径列表 (假设后缀是 .usd，你可以根据实际修改)
        object_paths = [(Path(self.folder_path) / f"{n}").as_posix() for n in unique_names]

        # 创建映射字典: { "name1": 0, "name2": 1, ... }
        name_to_idx = {name: i for i, name in enumerate(unique_names)}

        # 生成 proto_indices: 长度等于数据总数，每个元素是对应的模型索引
        proto_indices = [name_to_idx[n] for n in names]

        # 构建 data_list
        data_list = []
        for i in range(len(df)):
            data_list.append([x[i], y[i], z[i], 0.0, 0.0, 1.0])

        return object_paths, proto_indices, data_list


class DistributionUtils:
    """分发工具类，提供高性能的物体放置算法"""

    @staticmethod
    def ensure_prototypes_on_stage(stage, object_paths: List[str], proto_root_path: str) -> List[str]:
        """
        处理引用逻辑
        1. 文件资源 -> 创建代理并 AddReference。
        2. 舞台资源 -> MovePrim 到隐藏的 Prototypes 文件夹下。
        """
        stage_proto_paths = []

        # 确保原型根容器存在并隐藏
        if not stage.GetPrimAtPath(proto_root_path):
            UsdGeom.Scope.Define(stage, proto_root_path)
            #  “Prototype Invisibility Pattern”
            # 隐藏整个原型容器，这样原始物体就不会出现在场景中
            stage.GetPrimAtPath(proto_root_path).GetAttribute("visibility").Set("invisible")

        for i, path in enumerate(object_paths):
            is_external = "://" in path or os.path.exists(path)

            # 生成一个干净的名称
            base_name = os.path.basename(path.rstrip("/\\"))
            clean_name = re.sub(r'[^\w]', '_', os.path.splitext(base_name)[0])
            target_path = f"{proto_root_path}/{clean_name}"

            # 防止重名冲突
            if stage.GetPrimAtPath(target_path):
                target_path = f"{target_path}_{i}"

            if is_external:
                # --- 情况 A: 外部文件 -> 创建代理节点并引用 ---
                proto_prim = stage.DefinePrim(target_path, "Xform")
                proto_prim.GetReferences().AddReference(path)
                stage_proto_paths.append(target_path)
                carb.log_info(f"[Distribution] 文件已加载至原型库: {target_path}")

            else:
                # --- 情况 B: 已经是舞台 Prim -> 将其移动到原型库下 ---
                source_prim = stage.GetPrimAtPath(path)
                if source_prim.IsValid():
                    # 如果该 Prim 已经在我们的原型库里了，就不动它
                    if path.startswith(proto_root_path):
                        stage_proto_paths.append(path)
                    else:
                        # 使用 MovePrim 命令将物体挪进我们的“隐藏间”
                        # 这会改变物体的路径，但会保留它的所有属性和材质
                        omni.kit.commands.execute(
                            "MovePrim",
                            path_from=path,
                            path_to=target_path)
                        stage_proto_paths.append(target_path)
                        carb.log_info(f"[Distribution] 舞台 Prim 已移动至原型库: {target_path}")
                else:
                    carb.log_warn(f"[Distribution] 找不到路径对应的 Prim: {path}")
                    stage_proto_paths.append(path)

        return stage_proto_paths

    @staticmethod
    async def place_with_pointinstancer(
        object_path: Union[str, List[str]],
        data_list: list,
        proto_indices: List[int] = None,
        proto_root_path: str = "/World/Prototypes",
        instancer_path: str = "/World/PointInstancer"
    ):
        """
        方式一: PointInstancer (高性能，由外部指定模型索引)

        参数:
        - object_path: 模型路径列表
        - data_list: 位置和法线数据二维列表 [ [x,y,z,nx,ny,nz], ... ]
        - proto_indices: 整数列表，长度必须等于 data_list 长度。例如 [0, 2, 1, 0...]
        """
        # 如果传入的是字符串，则转换为列表
        if isinstance(object_path, str):
            object_path = [object_path]

        # 如果没传索引，默认全部指向第 0 个模型
        if proto_indices is None:
            proto_indices = [0] * len(data_list)

        # 获取舞台
        stage = omni.usd.get_context().get_stage()

        # 清理旧的PointInstancer节点
        if stage.GetPrimAtPath(instancer_path):
            omni.kit.commands.execute("DeletePrims", paths=[instancer_path])

        # 定义PointInstancer
        instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)

        # 确保原型在舞台上并获取它们的路径
        stage_proto_paths = DistributionUtils.ensure_prototypes_on_stage(
            stage, object_path, proto_root_path)

        # 将生成的舞台原型路径绑定给 PointInstancer
        prototypes_rel = instancer.GetPrototypesRel()
        for p_path in stage_proto_paths:
            prototypes_rel.AddTarget(Sdf.Path(p_path))

        # 4. 准备数据
        positions = []
        orientations = []
        scales = []

        for data in data_list:
            x, y, z, nx, ny, nz = map(float, data)
            # 位置
            positions.append(Gf.Vec3f(x, y, z))

            # 旋转 (法线对齐 + 随机自转)
            normal_vec = Gf.Vec3d(nx, ny, nz)
            align_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), normal_vec)
            yaw_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), random.uniform(0, 360))
            # 注意:PointInstancer 属性通常使用单精度
            quat_d = (yaw_rot * align_rot).GetQuat()
            orientations.append(Gf.Quath(quat_d))

            # 缩放 (这里只设置随机系数，它会自动乘到原型的 unitsResolve 上)
            s = random.uniform(0.9, 1.1)
            scales.append(Gf.Vec3f(s, s, s))

        # 5. 写入属性
        instancer.GetPositionsAttr().Set(Vt.Vec3fArray(positions))
        instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
        instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))
        instancer.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices))

        return instancer_path

    @staticmethod
    async def place_with_prim_instancing(
        object_path: Union[str, List[str]],
        data_list: list,
        proto_indices: List[int] = None,
        proto_root_path: str = "/World/Prototypes",
        instancer_path: str = "/World/PrimInstancer"
    ):
        """
        方式二: Prim Instancing (适合少量物体，可单独控制每一个 Prim)

        参数:
        - object_path: 模型路径列表
        - data_list: 位置数据列表 [ [x,y,z,nx,ny,nz], ... ]
        - proto_indices: 整数列表，长度必须等于 data_list 长度。例如 [0, 2, 1, 0...]
        """
        # 统一 object_path 为列表格式
        if isinstance(object_path, str):
            object_path = [object_path]
        stage = omni.usd.get_context().get_stage()

        # 如果没传索引，默认全部指向第 0 个模型
        if proto_indices is None:
            proto_indices = [0] * len(data_list)

        # 清理旧的节点
        if stage.GetPrimAtPath(instancer_path):
            omni.kit.commands.execute("DeletePrims", paths=[instancer_path])
        UsdGeom.Xform.Define(stage, instancer_path)

        # 注册原型：拿到一组已经在舞台上的prim路径
        stage_proto_paths = DistributionUtils.ensure_prototypes_on_stage(
            stage, object_path, proto_root_path)

        for i, data in enumerate(data_list):
            x, y, z, nx, ny, nz = map(float, data)

            # 获取当前位置对应的模型路径
            proto_index = proto_indices[i]
            target_stage_path = stage_proto_paths[proto_index]

            # 获取object的名字
            base_name = os.path.basename(target_stage_path.rstrip("/\\"))
            clean_name = os.path.splitext(base_name)[0]

            # 创建容器
            container_path = f"{instancer_path}/{clean_name}_{i}"
            container_prim = UsdGeom.Xform.Define(stage, container_path)

            # 设定位置
            xformable = UsdGeom.Xformable(container_prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

            # 设定随机旋转
            align_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(nx, ny, nz))
            yaw_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), random.uniform(0, 360))
            xformable.AddOrientOp().Set(Gf.Quatf((yaw_rot * align_rot).GetQuat()))

            # 设定随机缩放
            s = random.uniform(0.9, 1.1)
            xformable.AddScaleOp().Set(Gf.Vec3f(s, s, s))

            instance_path = f"{container_path}/Instance"
            instance_prim = stage.DefinePrim(instance_path, "Xform")

            # 添加reference的方法,使用InternalReference可以在stage上控制原型
            instance_prim.GetReferences().AddInternalReference(Sdf.Path(target_stage_path))

            # 开启实例化 (这是 Prim Instancing 的核心)
            instance_prim.SetInstanceable(True)

        return instancer_path
