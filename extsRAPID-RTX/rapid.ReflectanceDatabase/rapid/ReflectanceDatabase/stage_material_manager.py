import asyncio
from pxr import Usd, Gf, Sdf, UsdShade
import omni.usd
import omni.kit.commands
from typing import Tuple, Dict
import omni.replicator.core as rep

LOOKS_SCOPE = "/World/Looks"


async def updata_stage_materials(result_data: Dict):
    """
    获取窗口解析后的字典数据, 循环同步到 Stage
    参数 result_data 格式: {'leaf': {'ref': [0.1, ...], 'tra': [0.1, ...], 'display_color': [0.1,0.1,0.1]}, 'Name': {...}}
    """
    stage = omni.usd.get_context().get_stage()

    # 确保父级 Looks 路径存在
    if not stage.GetPrimAtPath(LOOKS_SCOPE):
        stage.DefinePrim(LOOKS_SCOPE, "Scope")

    # --- 循环处理字典中的每一条数据 ---
    for name, content in result_data.items():
        # 材质路径
        target_path = f"{LOOKS_SCOPE}/{name}"
        # 读取和转换RGB颜色数据格式为元组
        display_color = content.get('display_color', [])
        display_color_tuple = (display_color[0], display_color[1], display_color[2])

        # 通过透射率判断是否为叶片
        tran = content.get('tra', [])
        is_leaf = any(val > 0.001 for val in tran)
        tra_vec = display_color_tuple  # 假设透射率和RGB颜色相同

        # 判断 Stage 上是否已存在该材质
        prim = stage.GetPrimAtPath(target_path)
        if not prim:
            # A. 不存在：调用创建接口
            create_optical_material(name, display_color_tuple, tra_vec, is_leaf)
        else:
            # B. 已存在：直接更新 Shader 的属性（不需要重新创建整个材质）
            _update_existing_material_attributes(prim, display_color_tuple, tra_vec, is_leaf)

    prim = stage.GetPrimAtPath('/Replicator')
    if prim:
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()
        omni.kit.commands.execute('DeletePrims', paths=[Sdf.Path('/Replicator')], destructive=False)


def create_optical_material(name: str, ref: Tuple, tra: Tuple, is_leaf):
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
        on_created_fn=callback
    )


def _update_existing_material_attributes(mat_prim: Usd.Prim, ref: Tuple, tra: Tuple, is_leaf):
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
