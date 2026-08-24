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


class VegetationDistributor:
    def __init__(self, object_path, distribution_data, mode="PrimInstancing"):
        """
        Args:
            object_path [Dict]: 资产路径或者stage的prim路径,由 ensure_prototypes_on_stage判定处理
                结构如：{ "species_name":"usd_assest_path",...{
            List[Dict[str, Any]]: 每个树种一个字典的列表，结构如下：
                [
                    {
                        "species_name": "pine",          # 树种名称
                        "positions": np.ndarray,    # shape (M, 3)，每行 [x, y, z]
                        "normals": np.ndarray       # shape (M, 3)，每行 [nx, ny, nz]
                    },
                    ...
                ]
            dist_type: 分布类型 ("Random" / "Uniform")
            mode: 实例化模式 ("PointInstancer" 或 "PrimInstancing")
        Return:
        """
        self.object_path = object_path
        self.distribution_data = distribution_data
        self.mode = mode  # 保存模式

    def run(self):

        stage = omni.usd.get_context().get_stage()

        # 创建原型和实例化的文件夹路径
        proto_root_path = "/World/Prototypes"
        vegetation_root_path = "/World/Vegetation"
        if not stage.GetPrimAtPath(proto_root_path):
            proto_root = UsdGeom.Scope.Define(stage, proto_root_path)
            proto_root.GetPrim().GetAttribute("visibility").Set("invisible")
        if not stage.GetPrimAtPath(vegetation_root_path):
            UsdGeom.Scope.Define(stage, vegetation_root_path)

        results = []  # 存储返回信息
        if self.mode == "PrimInstancing":
            for species in self.distribution_data:
                # 获取每个树种的名字，usd模型文件路径信息
                species_name = species["species_name"]
                usd_asset_path = self.object_path[species_name]
                # 获取位置和法线信息
                merged = np.hstack((species["positions"], species["normals"]))  # shape: (M, 6)
                # 执行
                instancer_path = f"/World/Vegetation/{species_name}"
                result = DistributionUtils.place_with_prim_instancing(
                    species_name=species_name,
                    object_path=usd_asset_path,
                    data_list=merged,
                    instancer_path=instancer_path,
                    proto_root_path="/World/Prototypes",
                )
                results.append({"species_name": species_name,
                                "prototype_path": result,
                                "instancer_path": instancer_path,
                                "count": len(species["positions"])})
        # 根据模式选择分发方式
        elif self.mode == "PrimInstancing":
            DistributionUtils.place_with_pointinstancer(self.prim_path, self.distribution_data)
        return results


class DistributionUtils:
    """分发工具类，提供高性能的物体放置算法"""

    @staticmethod
    def ensure_prototypes_on_stage(stage, species_name: str, object_path: str, proto_root_path: str = "/World/Prototypes") -> str:
        """
        确保植被资源以隐藏原型形式存在, 并返回其stage路径。
        处理引用逻辑
        1. 文件资源 -> 创建代理并 AddReference。
        2. 舞台资源 -> MovePrim 到隐藏的 Prototypes 文件夹下。
        """
        prototype_path = f"{proto_root_path}/{species_name}"

        if stage.GetPrimAtPath(prototype_path):
            return prototype_path

        # --- 情况 A: 外部文件 -> 创建代理节点并引用 ---
        # 判断是磁盘路径还是舞台路径
        is_external = "://" in object_path or os.path.exists(object_path)
        if is_external:
            proto_prim = stage.DefinePrim(prototype_path, "Xform")
            proto_prim.GetReferences().AddReference(object_path)
            carb.log_info(f"[Distribution] 文件已加载至原型库: {prototype_path}")
            return prototype_path

        # --- 情况 B: 已经是舞台 Prim -> 将其移动到原型库下 ---
        source_prim = stage.GetPrimAtPath(object_path)
        if source_prim.IsValid():
            if object_path.startswith(proto_root_path):
                return object_path
            # 使用 MovePrim 命令将物体挪进我们的“隐藏间”
            # 这会改变物体的路径，但会保留它的所有属性和材质
            omni.kit.commands.execute("MovePrim", path_from=object_path, path_to=prototype_path)
            carb.log_info(f"[Chat RAPID Distribution] 舞台 Prim 已移动至原型库: {prototype_path}")
            return prototype_path

        carb.log_warn(f"[Chat RAPID Distributio] 找不到路径对应的 Prim: {object_path}")
        return None

    @staticmethod
    def place_with_pointinstancer(
        object_path: Union[str, List[str]],
        data_list: list,
        proto_indices: List[int] = None,
        proto_root_path: str = "/World/Prototypes",
        instancer_path: str = "/World/PointInstancer"):
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
    def place_with_prim_instancing(
        species_name: str,
        object_path: str,
        data_list: np.ndarray,
        instancer_path: str,
        proto_root_path: str = "/World/Prototypes",
    ):
        """
        方式二: Prim Instancing (适合少量物体，可单独控制每一个 Prim)

        参数:
        - object_path: 模型路径列表
        - data_list: 位置数据列表 [ [x,y,z,nx,ny,nz], ... ]
        - proto_indices: 整数列表，长度必须等于 data_list 长度。例如 [0, 2, 1, 0...]
        """
        stage = omni.usd.get_context().get_stage()

        # 清理旧的节点
        if stage.GetPrimAtPath(instancer_path):
            omni.kit.commands.execute("DeletePrims", paths=[instancer_path])
        UsdGeom.Xform.Define(stage, instancer_path)

        # 注册原型：拿到一组已经在舞台上的prim路径
        prototype_path = DistributionUtils.ensure_prototypes_on_stage(
            stage, species_name, object_path, proto_root_path)
        if not prototype_path:
            carb.log_error(f"[Distribution] 无法创建树种原型: {species_name}, asset={object_path}")

        for i, data in enumerate(data_list):
            x, y, z, nx, ny, nz = map(float, data)
            # 创建容器
            container_path = f"{instancer_path}/{species_name}_{i}"
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
            instance_prim.GetReferences().AddInternalReference(Sdf.Path(prototype_path))

            # 开启实例化 (这是 Prim Instancing 的核心)
            instance_prim.SetInstanceable(True)

        return prototype_path