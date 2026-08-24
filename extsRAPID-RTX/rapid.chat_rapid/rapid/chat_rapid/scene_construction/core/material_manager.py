import asyncio
from pxr import Usd, Gf, Sdf, UsdShade, UsdGeom
import pxr
import omni.usd
import omni.kit.commands
from typing import Tuple, Dict, List, Any
import omni.replicator.core as rep
import logging
logger = logging.getLogger(__name__)

LOOKS_SCOPE = "/World/Looks"


class MaterialAssigner:
    """Manage vegetation materials and bind them to prototype meshes."""

    def __init__(self, material_root: str = "/World/Materials"):
        """Initialize the material manager.

        Args:
            material_root: Root USD path used to store generated materials.

        Returns:
            None.
        """
        self.material_root = material_root

    async def apply_species_materials(self, material_RGB: Dict[str, Dict[str, List[float]]],
                                material_mesh_rules: Dict[str, Dict[str, List[str]]],
                                prototype_paths: Dict[str, str]):
        """主循环，将每个树种材质应用于所有植物原型.
        Args:
            material_RGB (Dict): 每个树种的材质的RGB信息.
                                Example:{'树种名':{ "foliage_color":[0.1,0.1,0.1],"trunk_color":[0.1,0.1,0.1] },...}
            material_mesh_rules: 每种树种的网格分类规则, 分叶片和树干.
                                Example:{'树种名':{"foliage": [], "trunk": []}....}
            prototype_paths: 从物种名称到stage上原型 USD 原始路径的映射.
                                Example:{species_name: prototype_path, ...}
        Returns:
            Dictionary containing material application results for each species.
        """
        stage = omni.usd.get_context().get_stage()
        results = {}

        # 循环，为每一个树种配置材质
        for species, prototype_path in prototype_paths.items():
            # 获取每一个树种的RGB值和mesh分类信息
            rgb_config = material_RGB.get(species, {})
            mesh_rules = material_mesh_rules.get(species, {})

            if not prototype_path:
                results[species] = {"status": "FAILED", "message": "Prototype path is empty."}
                error_msg = f"Prototype path is empty. species: {species}"
                logger.error(error_msg)
                continue

            # 开始执行
            results[species] = await self._apply_species_material(
                stage, species, prototype_path, rgb_config, mesh_rules)

        return results

    async def _apply_species_material(self, stage, species: str, prototype_path: str,
                                rgb_config: Dict[str, List[float]],
                                mesh_rules):
        """主要入口,创建物种材质并将其绑定到匹配的原型网格上.
        Args:
            stage: Current USD stage.
            species (str): Species name.
            prototype_path (str): USD prototype prim path.
            rgb_config Dict[str, List[float]]: Foliage and trunk RGB configuration.
            mesh_rules (Dict): Mesh name/path matching rules.
        Returns:
            Dictionary containing matched mesh and material information.
        """
        prototype = stage.GetPrimAtPath(prototype_path)
        if not prototype.IsValid():
            error_msg = f"Prototype not found: species: {species}"
            logger.error(error_msg)
            return {"status": "FAILED", "message": f"Prototype not found: {prototype_path}"}

        materials_path = {}  # 记录材质路径
        matched_meshes = []

        # 1.分别创建树叶和树干的材质
        for material_type in ("foliage", "trunk"):
            # 获取材质的RGB数值和材质名字
            color = rgb_config.get(f"{material_type}_color")
            material_name = f"{species}_{material_type}"
            # 判断是树叶还是树干材质
            if material_type == "foliage":
                is_foliage = True
                materials_path["foliage"] = f"{LOOKS_SCOPE}/{material_name}"
            elif material_type == "trunk":
                is_foliage = None
                materials_path["trunk"] = f"{LOOKS_SCOPE}/{material_name}"
            # 创建材质
            await MaterialManager.update_stage_materials(
                stage, is_foliage, material_name, color)

        # 2.绑定mesh和材质
        # 遍历路径下的子类，找出所有的mesh类型
        for mesh_prim in Usd.PrimRange(prototype):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            # 判断是叶子mesh还是树干mesh，返回str,"foliage"/"trunk"
            mesh_type = self._match_mesh_type(mesh_prim.GetName(), str(mesh_prim.GetPath()), mesh_rules)
            # 根据类型选择材质路径
            material_path = materials_path[mesh_type]
            # 绑定材质
            omni.kit.commands.execute('BindMaterial',
                material_path=material_path,
                prim_path=[mesh_prim.GetPath().pathString],
                strength=['weakerThanDescendants'],
                material_purpose='')

            # 记录返回
            matched_meshes.append({
                "mesh": str(mesh_prim.GetPath()),
                "type": mesh_type,
                "material": str(material_path)})

        return {
            "status": "PASS",
            "prototype": prototype_path,
            "materials": materials_path,
            "matched_meshes": matched_meshes,
        }

    def _match_mesh_type(self, mesh_name: str, mesh_path: str, mesh_rules):
        """Classify a mesh according to its name and USD path.判断是树叶mesh还是树干mesh
        Args:
            mesh_name: Mesh prim name.
            mesh_path: Full USD path of the mesh.
            mesh_rules: Classification keywords for foliage and trunk.
        Returns:
            "foliage", "trunk", or None when no rule matches.
        """
        search_text = f"{mesh_name} {mesh_path}".lower()

        for keyword in mesh_rules.get("foliage", []):
            if keyword.lower() in search_text:
                return "foliage"

        for keyword in mesh_rules.get("trunk", []):
            if keyword.lower() in search_text:
                return "trunk"

        return None


class MaterialManager:
    """Manage creation, update, and deletion of USD materials."""

    @staticmethod
    async def update_stage_materials(stage, is_foliage, material_name, color):
        """
        通用的材质创建接口
        获取窗口解析后的字典数据, 循环同步到 Stage
        参数 result_data 格式: {'leaf': {'ref': [0.1, ...], 'tra': [0.1, ...], 'display_color': [0.1,0.1,0.1]}, 'Name': {...}}
        """
        if not stage:
            stage = omni.usd.get_context().get_stage()

        # 确保父级 Looks 路径存在
        if not stage.GetPrimAtPath(LOOKS_SCOPE):
            stage.DefinePrim(LOOKS_SCOPE, "Scope")

        # 材质路径
        target_path = f"{LOOKS_SCOPE}/{material_name}"
        # 读取和转换RGB颜色数据格式为元组
        display_color_tuple = (color[0], color[1], color[2])

        # 判断 Stage 上是否已存在该材质
        prim = stage.GetPrimAtPath(target_path)
        if not prim:
            # A. 不存在：调用创建接口,displaycolor的反射率和透射率共用一个RGB值
            await MaterialManager.create_optical_material(material_name, display_color_tuple, display_color_tuple, is_foliage)
        else:
            # B. 已存在：直接更新 Shader 的属性（不需要重新创建整个材质）
            MaterialManager.update_existing_material_attributes(prim, display_color_tuple, display_color_tuple, is_foliage)

        # 确保 Kit 有机会完成 USD / MDL 更新
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        # 删除残留的rep
        prim = stage.GetPrimAtPath('/Replicator')
        if prim:
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
            omni.kit.commands.execute('DeletePrims', paths=[Sdf.Path('/Replicator')], destructive=False)

    async def create_optical_material(name: str, ref: Tuple, tra: Tuple, is_leaf):
        """执行 USD 命令创建材质结构"""
        async def set_shader_attributes(shader_prim: Usd.Prim, ref, tran):
            # 设定叶片和非叶片材质通用的漫反射权重和镜面反射权重
            shader_prim.CreateAttribute('inputs:diffuse_reflection_weight', Sdf.ValueTypeNames.Float).Set(1.0)
            shader_prim.CreateAttribute('inputs:specular_reflection_weight', Sdf.ValueTypeNames.Float).Set(0.0)

            # 具体属性
            if is_leaf:
                shader_prim.CreateAttribute('inputs:diffuse_reflection_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(ref[0]*2, ref[1]*2, ref[2]*2))
                shader_prim.CreateAttribute('inputs:enable_diffuse_transmission', Sdf.ValueTypeNames.Bool).Set(True)
                shader_prim.CreateAttribute('inputs:subsurface_weight', Sdf.ValueTypeNames.Float).Set(0.5)
                shader_prim.CreateAttribute('inputs:subsurface_transmission_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(tran[0]*2, tran[1]*2, tran[2]*2))
                shader_prim.CreateAttribute('inputs:thin_walled', Sdf.ValueTypeNames.Bool).Set(True)
            else:
                shader_prim.CreateAttribute('inputs:diffuse_reflection_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(ref[0], ref[1], ref[2]))

        def callback(prim):
            asyncio.ensure_future(set_shader_attributes(prim, ref, tra))

        omni.kit.commands.execute(
            'CreateAndBindMdlMaterialFromLibrary',
            mdl_name='OmniSurface.mdl',
            mtl_name='OmniSurface',
            prim_name=name,
            on_created_fn=callback)

    @staticmethod
    def update_existing_material_attributes(mat_prim: Usd.Prim, ref: Tuple, tra: Tuple, is_leaf):
        """
        内部函数：当材质已存在时，只更新它的 Shader 属性，避免重复调用创建命令
        """
        # 找到材质下的 Shader Prim (通常是由命令创建的名为 Shader 的子 Prim)
        shader_prim = None
        for child in mat_prim.GetChildren():
            if child.IsA(UsdShade.Shader):
                shader_prim = child
                break

        shader_prim = rep.get.prims(path_pattern=str(shader_prim.GetPath()))
        if is_leaf:
            # 叶片：更新反射率和透射率（2倍增强逻辑）
            with shader_prim:
                rep.modify.attribute(
                                    name="inputs:diffuse_reflection_color",
                                    value=rep.distribution.sequence([(ref[0]*2, ref[1]*2, ref[2]*2)]),
                                    attribute_type="color3f"
                                )
                rep.modify.attribute(
                                    name="inputs:subsurface_transmission_color",
                                    value=rep.distribution.sequence([(tra[0]*2, tra[1]*2, tra[2]*2)]),
                                    attribute_type="color3f"
                                )
        else:
            # 其他：更新反射率，关闭透射
            with shader_prim:
                rep.modify.attribute(
                                    name="inputs:diffuse_reflection_color",
                                    value=rep.distribution.sequence([(ref[0], ref[1], ref[2])]),
                                    attribute_type="color3f"
                                )

    @staticmethod
    def delete_stage_material(name: str):
        material_stage_path = '/World/Looks/' + name

        # 获取当前 Stage
        stage = omni.usd.get_context().get_stage()
        # 检查 Prim 是否存在
        prim = stage.GetPrimAtPath(material_stage_path)
        if not prim.IsValid():
            # 如果不存在则直接返回
            print(f"Material not found: {material_stage_path}")
            return

        # 删除
        omni.kit.commands.execute(
            'DeletePrims',
            paths=[Sdf.Path(material_stage_path)],
            destructive=False)
