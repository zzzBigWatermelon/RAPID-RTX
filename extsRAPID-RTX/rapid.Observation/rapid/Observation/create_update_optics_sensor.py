from pxr import UsdGeom, Sdf, Gf
import omni.kit.commands
import omni.usd
import omni.kit.notification_manager as nm
import math
# 自定义功能
from rapid.Utility.sensor_params import set_prim_attributes
from rapid.Utility.calculate_sampling_waypoints import calculate_constant_altitude_sampling_waypoints  # 计算航线位置
from rapid.Utility.calculate_sampling_waypoints import calculate_footprint_from_camera  # 计算视野覆盖范围
from rapid.Utility.calculate_sampling_waypoints import calculate_semi_circular_sampling_waypoints
from rapid.Utility.calculate_sampling_waypoints import calculate_omnidirectional_sampling_waypoints
from .visualize_waypoints import visualize_waypoints


def create_optics_sensor(data):
    '''
    在舞台上创建相机,并按照窗口参数写入模拟参数

    Args:
        data: 说明

    Returns:
        camera_path: 说明
    '''
    # 获取基础属性，下面的key与observation_windows.py中_init_models中的相同
    stage = omni.usd.get_context().get_stage()
    path = data['sensor_stage_path']
    sensor_type = data['sensor_type']
    observation_type = data['optical_observation_type']
    sensor_pixels = data['optical_sensor_pixels']

    # 创建并初始化相机 Prim
    camera = UsdGeom.Camera.Define(stage, path)
    camera_prim = camera.GetPrim()

    # ---------------------------计算和设定传感器参数-----------------------------
    # 设置标准相机属性
    proj_type = "perspective" if sensor_type != "Orthographic" else "orthographic"
    camera.GetProjectionAttr().Set(proj_type)

    # 设置的传感器旋转属性，从位置朝向目标（目前只针对单次采样）、顺便添加tran和rotate属性
    handling_coordinates_rotations(camera, data['single_sampling_sensor_position'], data['single_sampling_observation_center'])
    # 计算传感器光学参数——传感器物理宽度和焦距(函数内部已经更新对应属性)
    horizontal_aperture, focal_length = calculate_focal_length(camera, sensor_type, sensor_pixels, data['perspective_sensor_fov'], data['orthographic_sensor_extent'])

    # ---------------------------写入自定义传感器参数-----------------------------
    # 写入基础属性
    set_prim_attributes(camera_prim, "Base Attribute", data)
    # 写入传感器参数 (如视角像素、FOV)
    set_prim_attributes(camera_prim, sensor_type, data)
    # 写入采样模式参数 (如恒定高度参数、半圆采样参数)
    set_prim_attributes(camera_prim, observation_type, data)

    # 在舞台上创建观测航线的可视化
    create_waypoints_visualization(data, observation_type, camera_prim, horizontal_aperture, focal_length, sensor_pixels, sensor_type)

    # 创建完传感器的提示
    nm.post_notification(
        "Camera created successfully. The corresponding sensor can be found in the stage tab.",  # 这里可以改成你创建完相机的提示内容
        status=nm.NotificationStatus.INFO,  # 蓝色提醒
        duration=5
    )
    return path


def handling_coordinates_rotations(camera_prim, position, target):
    '''Translate和Rotate会更加明了直接'''
    # 清理旧的所有变换操作，确保属性栏干净
    camera_prim.GetXformOpOrderAttr().Set([])

    # 设置位移 (Translate)
    translate_op = camera_prim.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*position))

    # B. 计算并设置旋转 (Rotate)
    eye = Gf.Vec3d(*position)
    at = Gf.Vec3d(*target)
    # 根据场景向上轴设置。如果看不到，尝试 (0, 1, 0)
    up = Gf.Vec3d(0, 0, 1)

    # 使用 LookAt 计算矩阵，然后分解出欧拉角 (Euler Angles)
    lookat_mat = Gf.Matrix4d().SetLookAt(eye, at, up)
    world_mat = lookat_mat.GetInverse()  # 获取相机的世界变换矩阵

    # 分解旋转
    rotation = world_mat.ExtractRotation()
    # 转换为常用的 XYZ 欧拉角 (度数)
    euler = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))

    rotate_op = camera_prim.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(euler))  # USD 的旋转通常使用 float3


def calculate_focal_length(camera, sensor_type, sensor_pixels, sensor_diagFOV, extent):
    '''
    用户输入的相机参数转omniverse相机需要的参数
    透视相机用户输入像素大小和对角线视场角 -> omniverse的焦距和传感器宽度
    正射相机用户观测范围 -> omniverse的传感器宽度,焦距在正射模式中没用,aperture的单位是世界单位的十分之一

    Args:
        self._sensor_pixels: [weight, height],传感器宽高像素(pixels),;
        self._sensor_diagFOV: 对角线视场角(°);.

    Returns:
        焦距(mm);
    '''
    horizontal_aperture = 0
    focal_length = 0
    # 计算透视相机的传感器宽度和焦距
    if sensor_type == 'Perspective':
        aspect_ratio = sensor_pixels[0] / sensor_pixels[1]

        # 计算焦距与传感器宽度的比值, f/W = (对角线/宽度) / (2 * tan(FOV/2))
        # 根据宽高比计算对角线长度与宽度的比值,传感器对角线/宽度 = √(1 + (1/aspect_ratio)²)
        diag_to_width_ratio = math.sqrt(1 + (1 / aspect_ratio) ** 2)
        # 计算焦距与宽度的比值
        tan_half_fov = math.tan(math.radians(sensor_diagFOV) / 2)
        # f/W = (对角线/宽度) / (2 * tan(FOV/2))
        focal_to_width_ratio = diag_to_width_ratio / (2 * tan_half_fov)

        # 计算焦距和传感器宽度，没有单位，只要focal_to_width_ratio正确即可
        horizontal_aperture = 36.0   # 指定传感器宽度可以是 36、1、100，全都等价
        focal_length = focal_to_width_ratio * horizontal_aperture

    # 计算正射相机的传感器宽度horizontal_aperture，焦距在正射模式中没用
    # 正射相机的horizontal_aperture就是观测范围,但是aperture的单位是世界单位的十分之一,所以*10
    elif sensor_type == 'Orthographic':
        horizontal_aperture = extent*10
        focal_length = 1  # 焦距在正射模式中没用，随意设定

    # 设定相机属性
    camera.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
    camera.GetFocalLengthAttr().Set(focal_length)

    return horizontal_aperture, focal_length


def create_waypoints_visualization(data, obs_type, cmaera_prim, horizontal_aperture, focal_length, pixels, sensor_type):
    # 存储观测位置和视野范围
    observation_position = []
    observation_center = None
    ground_footprint = []

    # 获取观测模式
    if obs_type == 'Single Sampling':
        camera_geom = UsdGeom.Camera(cmaera_prim)
        camera_translate_value = camera_geom.GetTranslateOp().Get()
        # 单次观测不做位置计算,只返回地面范围，start_point=0开启单次观测计算
        ground_footprint = calculate_footprint_from_camera(focal_length, horizontal_aperture, pixels, camera_translate_value[2])
        observation_position.append(camera_translate_value)

    elif obs_type == 'Constant Altitude Sampling':
        observation_position, ground_footprint = calculate_constant_altitude_sampling_waypoints(
            data['constant_altitude_sampling_start_point'], data['constant_altitude_sampling_end_point'], data['constant_altitude_sampling_flight_altitude'],
            pixels, focal_length, horizontal_aperture, data['constant_altitude_sampling_forward_and_side_overlap']
        )

    elif obs_type == 'Semi-circular Sampling':
        observation_position, ground_footprint = calculate_semi_circular_sampling_waypoints(
            data['semicircular_sampling_distance'], data['semicircular_sampling_view_azimuth'], data['semicircular_sampling_zenith_range'],
            data['semicircular_sampling_zenith_step'], data['semicircular_sampling_observation_center']
        )
        observation_center = data['semicircular_sampling_observation_center']

    elif obs_type == 'Omnidirectional Sampling':
        observation_position, ground_footprint = calculate_omnidirectional_sampling_waypoints(
            data['omnidirectional_sampling_distance'], data['omnidirectional_sampling_view_zenith'],
            data['omnidirectional_sampling_azimuth_step'], data['omnidirectional_sampling_observation_center']
        )
        observation_center = data['omnidirectional_sampling_observation_center']

    visualize_waypoints(observation_position, sensor_type, observation_center)
