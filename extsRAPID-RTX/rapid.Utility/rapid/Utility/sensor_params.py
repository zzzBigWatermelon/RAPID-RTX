from pxr import Sdf, Gf, UsdGeom, Vt


# 定义不同观测模式对应的属性： { 模式名: { 属性路径: (数据类型，数据字典中的键名) } }
# 模式名：是传感器类型和观测模式类型，对应observation_windows.py中_init_models
# 属性路径：usd属性的路径名，通过CreateAttribute创建，后续通过GetAttribute获取
# 数据类型：usd数据类型
# 数据类型： set_prim_attributes参数data数据字典中的键名，与observation_windows.py中_init_models对应获取具体属性数值
OBSERVATION_CONFIG = {
    "Base Attribute": {  # 一般属性
        "rapid:sensor:sensor_stage_path": (Sdf.ValueTypeNames.String, "sensor_stage_path"),
        "rapid:sensor:is_rapid_sensor": (Sdf.ValueTypeNames.Bool, "is_rapid_sensor"),
        "rapid:sensor:sensor_type": (Sdf.ValueTypeNames.String, "sensor_type"),
        "rapid:sensor:optical_observation_type": (Sdf.ValueTypeNames.String, "optical_observation_type"),
    },
    "Perspective": {  # 透视相机
        "rapid:sensor:optical_sensor_pixels": (Sdf.ValueTypeNames.Float2, "optical_sensor_pixels"),
        "rapid:sensor:perspective_sensor_fov": (Sdf.ValueTypeNames.Float, "perspective_sensor_fov"),
    },
    "Orthographic": {  # 正交相机
        "rapid:sensor:optical_sensor_pixels": (Sdf.ValueTypeNames.Float2, "optical_sensor_pixels"),
        "rapid:sensor:orthographic_sensor_extent": (Sdf.ValueTypeNames.Float, "orthographic_sensor_extent"),
    },
    "Single Sampling": {
        "rapid:sensor:single_sampling:sensor_position": (Sdf.ValueTypeNames.Float3, "single_sampling_sensor_position"),
        "rapid:sensor:single_sampling:observation_center": (Sdf.ValueTypeNames.Float3, "single_sampling_observation_center"),
    },
    "Constant Altitude Sampling": {
        "rapid:sensor:constant_altitude_sampling:flight_altitude": (Sdf.ValueTypeNames.Float, "constant_altitude_sampling_flight_altitude"),
        "rapid:sensor:constant_altitude_sampling:flight_speed": (Sdf.ValueTypeNames.Float, "constant_altitude_sampling_flight_speed"),
        "rapid:sensor:constant_altitude_sampling:start_point": (Sdf.ValueTypeNames.Float2, "constant_altitude_sampling_start_point"),
        "rapid:sensor:constant_altitude_sampling:end_point": (Sdf.ValueTypeNames.Float2, "constant_altitude_sampling_end_point"),
        "rapid:sensor:constant_altitude_sampling:forward_and_side_overlap": (Sdf.ValueTypeNames.Float2, "constant_altitude_sampling_forward_and_side_overlap"),
    },
    "Semi-circular Sampling": {
        "rapid:sensor:semicircular_sampling:observation_center": (Sdf.ValueTypeNames.Float3, "semicircular_sampling_observation_center"),
        "rapid:sensor:semicircular_sampling:distance": (Sdf.ValueTypeNames.Float, "semicircular_sampling_distance"),
        "rapid:sensor:semicircular_sampling:view_azimuth": (Sdf.ValueTypeNames.Float, "semicircular_sampling_view_azimuth"),
        "rapid:sensor:semicircular_sampling:zenith_range": (Sdf.ValueTypeNames.Float2, "semicircular_sampling_zenith_range"),
        "rapid:sensor:semicircular_sampling:zenith_step": (Sdf.ValueTypeNames.Float, "semicircular_sampling_zenith_step"),
    },
    "Omnidirectional Sampling": {
        "rapid:sensor:semicircular_sampling:observation_center": (Sdf.ValueTypeNames.Float3, "omnidirectional_sampling_observation_center"),
        "rapid:sensor:semicircular_sampling:distance": (Sdf.ValueTypeNames.Float, "omnidirectional_sampling_distance"),
        "rapid:sensor:semicircular_sampling:view_zenith": (Sdf.ValueTypeNames.FloatArray, "omnidirectional_sampling_view_zenith"),
        "rapid:sensor:semicircular_sampling:azimuth_range": (Sdf.ValueTypeNames.Float2, "omnidirectional_sampling_azimuth_range"),
        "rapid:sensor:semicircular_sampling:azimuth_step": (Sdf.ValueTypeNames.Float, "omnidirectional_sampling_azimuth_step"),
    },
    "Airborne LiDAR": {
        "rapid:sensor:airborne_LiDAR:fov": (Sdf.ValueTypeNames.Float, "airborne_LiDAR_fov"),
        "rapid:sensor:airborne_LiDAR:angle_resolution": (Sdf.ValueTypeNames.Float, "airborne_LiDAR_angle_resolution"),
        "rapid:sensor:airborne_LiDAR:scan_rate": (Sdf.ValueTypeNames.Float, "airborne_LiDAR_scan_rate"),
        "rapid:sensor:airborne_LiDAR:start_point": (Sdf.ValueTypeNames.Float2, "airborne_LiDAR_start_point"),
        "rapid:sensor:airborne_LiDAR:end_point": (Sdf.ValueTypeNames.Float2, "airborne_LiDAR_end_point"),
        "rapid:sensor:airborne_LiDAR:flight_altitude": (Sdf.ValueTypeNames.Float, "airborne_LiDAR_flight_altitude"),
        "rapid:sensor:airborne_LiDAR:flight_speed": (Sdf.ValueTypeNames.Float, "airborne_LiDAR_flight_speed"),
        "rapid:sensor:airborne_LiDAR:strip_overlap": (Sdf.ValueTypeNames.Float, "airborne_LiDAR_strip_overlap"),
    },
    "Terrestrial LiDAR": {
        "rapid:sensor:terrestrial_LiDAR:min_zenith_angle": (Sdf.ValueTypeNames.Float, "terrestrial_LiDAR_min_zenith_angle"),
        "rapid:sensor:terrestrial_LiDAR:max_zenith_angle": (Sdf.ValueTypeNames.Float, "terrestrial_LiDAR_max_zenith_angle"),
        "rapid:sensor:terrestrial_LiDAR:zenith_angle_resolution": (Sdf.ValueTypeNames.Float, "terrestrial_LiDAR_zenith_angle_resolution"),
        "rapid:sensor:terrestrial_LiDAR:azimuth_angle_resolution": (Sdf.ValueTypeNames.Float, "terrestrial_LiDAR_azimuth_angle_resolution"),
        "rapid:sensor:terrestrial_LiDAR:sampling_frequency": (Sdf.ValueTypeNames.Float, "terrestrial_LiDAR_sampling_frequency"),
        "rapid:sensor:terrestrial_LiDAR:position": (Sdf.ValueTypeNames.Float3, "terrestrial_LiDAR_position"),

    },
    "Spaceborne LiDAR (Waveform LiDAR Data)": {
        "rapid:sensor:spaceborne_Lidar:footprint_width": (Sdf.ValueTypeNames.Float, "spaceborne_LiDAR_footprint_width"),
        "rapid:sensor:spaceborne_Lidar:system_pulse_width": (Sdf.ValueTypeNames.Float, "spaceborne_LiDAR_system_pulse_width"),
        "rapid:sensor:spaceborne_Lidar:vertical_bin_size": (Sdf.ValueTypeNames.Float, "spaceborne_LiDAR_vertical_bin_size"),
        "rapid:sensor:spaceborne_Lidar:footprint_center": (Sdf.ValueTypeNames.Float2, "spaceborne_LiDAR_footprint_center"),
    },
    # 以后增加新模式写在这里
}


def set_prim_attributes(prim, config_key, data):
    """
    根据 CONFIG 配置表，将 data 中的数据安全地写入 Prim 属性
    """
    if config_key not in OBSERVATION_CONFIG:
        return

    for attr_path, (attr_type, data_key) in OBSERVATION_CONFIG[config_key].items():
        val = data.get(data_key)
        if val is None:
            continue

        attr = prim.CreateAttribute(attr_path, attr_type)
        # 根据类型自动包装 Gf 数据类型
        if attr_type == Sdf.ValueTypeNames.Float2:
            attr.Set(Gf.Vec2f(*val))
        elif attr_type == Sdf.ValueTypeNames.Int2:
            attr.Set(Gf.Vec2i(*val))
        elif attr_type == Sdf.ValueTypeNames.Float3:
            attr.Set(Gf.Vec3f(*val))
        elif attr_type == Sdf.ValueTypeNames.FloatArray:
            attr.Set(Vt.FloatArray(val))
        else:
            attr.Set(val)


def get_prim_attributes(prim, config_key, data_out):
    """
    通用读取函数：根据 CONFIG 将 Prim 属性读入字典
    """
    if config_key not in OBSERVATION_CONFIG:
        return data_out

    mode_attrs = OBSERVATION_CONFIG[config_key]
    for attr_path, (attr_type, data_key) in mode_attrs.items():
        attr = prim.GetAttribute(attr_path)
        if not attr.IsValid():
            continue

        val = attr.Get()
        if val is None:
            continue

        # 通用类型转换
        # 只要是 Gf 的向量类（Vec2f, Vec3d, Quat 等），它们通常都是可迭代的
        if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
            data_out[data_key] = list(val)
        else:
            data_out[data_key] = val

    return data_out
