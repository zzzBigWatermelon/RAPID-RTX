'''这里不创建实际的LiDAR传感器,只创建LiDAR载体的Xform,然后使用debugDraw可视化
将所有的模拟参数写到载体上,给LiDAR模拟扩展传递LiDAR载体的stage路径
'''
from pxr import UsdGeom
import omni.kit.commands
import omni.usd
import omni.kit.notification_manager as nm
# 自定义功能
from rapid.Utility.sensor_params import set_prim_attributes
from rapid.Utility.calculate_sampling_waypoints import calculate_airborne_LiDAR_waypoints  # 计算航线位置
from .visualize_waypoints import visualize_waypoints, spaceborne_LiDAR_visualize


async def create_LiDAR_carrier(data):
    '''
    在舞台上创建LiDAR载体的Xform,并按照窗口参数写入模拟参数

    Args:
        data: 说明

    Returns:
        camera_path: 说明
    '''

    stage = omni.usd.get_context().get_stage()
    lidar_carrier_path = data['sensor_stage_path']
    sensor_type = data["sensor_type"]

    omni.kit.commands.execute('CreatePrimWithDefaultXform', prim_type='Xform', prim_path=lidar_carrier_path)
    lidar_carrier_prim = stage.GetPrimAtPath(lidar_carrier_path)
    xformable = UsdGeom.Xformable(lidar_carrier_prim)  # 添加Xformable移动功能

    # ---------------------------写入自定义传感器参数-----------------------------
    # 写入基础属性
    set_prim_attributes(lidar_carrier_prim, "Base Attribute", data)
    # 写入传感器参数
    set_prim_attributes(lidar_carrier_prim, sensor_type, data)

    # 计算航线的起始点
    if sensor_type == "Airborne LiDAR":
        waypoints_pairs = calculate_airborne_LiDAR_waypoints(data["airborne_LiDAR_start_point"], data["airborne_LiDAR_end_point"],
                                                             data["airborne_LiDAR_flight_altitude"], data["airborne_LiDAR_strip_overlap"], data["airborne_LiDAR_fov"])
        waypoints_list = [point for pair in waypoints_pairs for point in pair]
        # 可视化航线
        visualize_waypoints(waypoints_list, sensor_type)

    elif sensor_type == "Terrestrial LiDAR":
        waypoints = data["terrestrial_LiDAR_position"]
        waypoints_list.append((waypoints[0], waypoints[1], waypoints[2]))
        # 可视化位置
        visualize_waypoints(waypoints_list, sensor_type)

    elif sensor_type == "Spaceborne LiDAR (Waveform LiDAR Data)":
        # 获取光斑中心和直径参数
        footprint_center = data["spaceborne_LiDAR_footprint_center"]
        footprint_width = data["spaceborne_LiDAR_footprint_width"]
        # 可视化光斑的范围
        spaceborne_LiDAR_visualize(footprint_center, footprint_width)

    # 创建完传感器的提示
    nm.post_notification(
        "LiDAR created successfully. The corresponding sensor can be found in the stage.",  # 这里可以改成你创建完相机的提示内容
        status=nm.NotificationStatus.INFO,  # 蓝色提醒
        duration=5
    )
