import omni.kit.commands
from pxr import Sdf, Gf, UsdGeom


def _cteate_simple_terrain(terrain_extent):
    # 使用内置命令创建一个平面 Mesh
    plane_prim_path = '/World/Simple_Plane'
    omni.kit.commands.execute(
        'CreateMeshPrimWithDefaultXform',
        prim_type='Plane',
        prim_path=plane_prim_path
    )
    # 2. 获取 Stage 和 Prim
    stage = omni.usd.get_context().get_stage()
    plane_mesh = stage.GetPrimAtPath(plane_prim_path)
    # 3. 使用 UsdGeom 操作缩放
    xformable = UsdGeom.Xformable(plane_mesh)
    # 获取或创建 scale 操作符
    scale_op = xformable.GetScaleOp()
    if not scale_op:
        scale_op = xformable.AddScaleOp()

    # 设置缩放值为 (宽, 高, 厚度)
    scale_op.Set(Gf.Vec3f(terrain_extent[0], terrain_extent[1], 1.0))
