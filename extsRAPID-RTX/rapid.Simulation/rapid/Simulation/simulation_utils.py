import omni.replicator.core as rep
from itertools import zip_longest
from typing import Dict
import omni.usd
import carb
from pxr import Gf, UsdGeom


class MaterialUtils:
    '''
    '''
    @staticmethod
    def updata_stage_materials(result_data: Dict):
        """
        获取窗口解析后的字典数据, 循环同步到 Stage
        参数 result_data 格式: {'leaf': {'ref': [0.1, ...], 'tra': [0.1, ...], 'display_color': [0.1,0.1,0.1]}, 'Name': {...}}
        """
        # 处理原始反射率字典数据至shader的三波段反射投射数值
        reflectance_data, max_tuple_count = MaterialUtils.processing_raw_ref_dic(result_data)

        # 材质路径
        stage = omni.usd.get_context().get_stage()
        LOOKS_SCOPE = "/World/Looks"
        # --- 循环处理字典中的每一条数据 ---
        for name, content in reflectance_data.items():
            # 材质路径
            target_shader_path = f"{LOOKS_SCOPE}/{name}/Shader"

            # 判断 Stage 上是否已存在该材质
            prim = stage.GetPrimAtPath(target_shader_path)
            if not prim:
                carb.log_warn(f"Material does not exist: {target_shader_path}")
            else:
                # B. 已存在：直接更新 Shader 的属性（不需要重新创建整个材质）
                MaterialUtils.modifiy_material_attributes(target_shader_path, content)

        return max_tuple_count

    @staticmethod
    def processing_raw_ref_dic(reflectance_data, leaf_threshold=0.001):
        """
        处理反射/透射数据: 将叶片材质的'ref'和'tra'列表中的每个数值乘以2, 非叶片：保持原数值直接分组
        每三个值分为一组 (R, G, B)，不足三个的补 0, 返回一个新字典,可直接用于反射率材质修改语法

        参数:
            原始反射率字典 (dict):{'leaf': {'ref': [0.1,0.2,0.3...], 'tra': [0.1,0.2,0.3, ...], }, 'Name': {...}}
        返回:
            新字典 (dict):{'leaf': {'ref': [(0.1,0.2,0.3), ...], 'tra': [[(0.1,0.2,0.3), ...], }, 'Name': {...}}
        """
        processed = {}
        max_tuple_count = 0  # 记录最大元组数量
        for key, value in reflectance_data.items():
            entry = {}

            # 获取原始列表
            ref_raw = value.get('ref', [])
            tra_raw = value.get('tra', [])

            # 判断是否为叶片：tra 非空且存在任意值 > leaf_threshold
            is_leaf = bool(tra_raw) and any(v > leaf_threshold for v in tra_raw)
            # 确定缩放因子
            scale = 2.0 if is_leaf else 1.0

            # ----- 处理 ref -----
            if ref_raw:
                ref_scaled = [x * scale for x in ref_raw]
                args = [iter(ref_scaled)] * 3
                entry['ref'] = list(zip_longest(*args, fillvalue=0))

            # ----- 处理 tra -----
            # 只有当 tra 非空时才处理（无论是否为叶片）
            if tra_raw:
                tra_scaled = [x * scale for x in tra_raw]
                args = [iter(tra_scaled)] * 3
                entry['tra'] = list(zip_longest(*args, fillvalue=0))

            processed[key] = entry

            # 更新最大元组数量
            current_count = len(entry['tra'])
            if current_count > max_tuple_count:
                max_tuple_count = current_count
        return processed, max_tuple_count

    @staticmethod
    def modifiy_material_attributes(target_shader_path, reflectance_data):
        """
        修改材质的反射率
        参数:
            target_shader_path: 材质的shader路径
            reflectance_data: 数据字典 (dict):{'leaf': {'ref': [(0.1,0.2,0.3), ...], 'tra': [[(0.1,0.2,0.3), ...], }, 'Name': {...}}
        """
        # 判断是否为叶片类（根据透射率）
        is_leaf = any(x > 0.001 for tup in reflectance_data['tra'] for x in tup)
        shader_prim = rep.get.prims(path_pattern=target_shader_path)
        with rep.trigger.on_custom_event(event_name="change_color"):
            if is_leaf:
                # 叶片：更新反射率和透射率
                with shader_prim:
                    rep.modify.attribute(
                                        name="inputs:diffuse_reflection_color",
                                        value=rep.distribution.sequence(reflectance_data['ref']),
                                        attribute_type="color3f"
                                    )
                    rep.modify.attribute(
                                        name="inputs:subsurface_transmission_color",
                                        value=rep.distribution.sequence(reflectance_data['tra']),
                                        attribute_type="color3f"
                                    )
            else:
                # 其他：更新反射率，关闭透射
                with shader_prim:
                    rep.modify.attribute(
                                        name="inputs:diffuse_reflection_color",
                                        value=rep.distribution.sequence(reflectance_data['ref']),
                                        attribute_type="color3f"
                                    )


class CameraUtils:
    @staticmethod
    def set_camera_pose_lookat_quat(stage, camera_prim_path, eye, target, up=(0, 0, 1)):
        """
        自适应处理奇异点的 LookAt 函数
        """
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        if not camera_prim:
            camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path).GetPrim()

        eye_vec = Gf.Vec3d(eye)
        target_vec = Gf.Vec3d(target)

        # 1. 计算视线方向
        forward = (target_vec - eye_vec).GetNormalized()
        up_vec = Gf.Vec3d(up).GetNormalized()

        # 2. 核心修复：检查视线是否与 up 向量平行
        # 如果视线和 up 向量的点积接近 1 或 -1，说明平行
        if abs(Gf.Dot(forward, up_vec)) > 0.999:
            # 如果垂直向下/向上看 Z 轴，则切换临时 up 向量为 Y 轴
            up_vec = Gf.Vec3d(0, 1, 0) if abs(up_vec[2]) > 0.999 else Gf.Vec3d(0, 0, 1)

        # 3. 计算变换矩阵
        look_at_matrix = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, up_vec)
        transform_matrix = look_at_matrix.GetInverse()

        # 4. 提取位姿
        translate = transform_matrix.ExtractTranslation()
        quat_d = transform_matrix.ExtractRotationQuat()

        # 转换为 float 兼容 Isaac Sim
        quat_f = Gf.Quatf(float(quat_d.GetReal()), Gf.Vec3f(quat_d.GetImaginary()))

        # 5. 应用到 USD
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()

        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(translate)

        orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        orient_op.Set(quat_f)
