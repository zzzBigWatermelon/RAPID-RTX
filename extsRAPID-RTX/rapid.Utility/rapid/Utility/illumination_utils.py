import math
from datetime import datetime, timedelta, timezone
import numpy as np
import omni.usd
from pxr import Usd, Gf, UsdGeom, UsdLux
from isaacsim.util.debug_draw import _debug_draw


class IlluminationUtils:

    @staticmethod
    def create_light(sun_light_path='/Light/Sun_Light', sky_light_path='/Light/Sky_Light',
                     sun_light_radius=100, sun_light_intensity=8000000000.0,
                     sky_light_intensity=150):
        '''
        创建直射光(DiskLight)和穹顶光(DomeLight)，并统一放在 /Light 父节点下

        Args:
            sun_light_path: 以 /Light/ 开头
            sky_light_path: 以 /Light/ 开头
            sun_light_radius: 直射光半径
            sun_light_intensity: 直射光强度
            sky_light_intensity: 穹顶光强度
        Returns:
            sun_light_prim: 返回太阳光的 Prim 对象
        '''
        stage = omni.usd.get_context().get_stage()

        # 1. 明确创建父类 Xform 节点 /Light
        # 使用 UsdGeom.Xform.Define，如果已存在则直接获取，不存在则创建
        light_parent_path = "/Light"
        if not stage.GetPrimAtPath(light_parent_path):
            UsdGeom.Xform.Define(stage, light_parent_path)

        # 2. 创建并配置 SunLight (DiskLight)
        # 即使传入的路径不是以 /Light 开头，USD 也会根据路径自动创建层级
        sun_light = UsdLux.DiskLight.Define(stage, sun_light_path)
        sun_light.GetIntensityAttr().Set(sun_light_intensity)
        sun_light.GetRadiusAttr().Set(sun_light_radius)
        # 开启 Normalize光照不随半径增大而变亮（只改变阴影虚实）
        # sun_light.GetNormalizeAttr().Set(True)

        # 3. 创建并配置 DomeLight (SkyLight)
        sky_light = UsdLux.DomeLight.Define(stage, sky_light_path)
        sky_light.GetIntensityAttr().Set(sky_light_intensity)

        return sun_light.GetPrim()

    @staticmethod
    def setup_sun_light(light_path, zenith_deg, azimuth_deg, light_center=(0.0, 0.0, 0.0), distance=100000.0):
        """
        使用极坐标控制 SunLight 的几何属性，并指定光照中心

        Args:
            light_path: USD 路径
            zenith_deg: 天顶角 (0-90)
            azimuth_deg: 方位角 (0-360)
            light_center: 光照的目标中心点坐标 (x, y, z)
            distance: 灯光距离中心点的距离
        """
        stage = omni.usd.get_context().get_stage()
        # 获取 Prim 对象
        prim = stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            print(f"Warning: {light_path} is not valid.")
            return

        # 转换为 Gf.Vec3d 方便后续数学运算
        target_center = Gf.Vec3d(light_center)

        # 1. 角度转弧度并计算相对于中心点的偏移位置 (Local Offset)
        zenith_rad = np.radians(zenith_deg)
        azimuth_rad = np.radians(azimuth_deg)

        off_x = distance * np.sin(zenith_rad) * np.cos(azimuth_rad)
        off_y = distance * np.sin(zenith_rad) * np.sin(azimuth_rad)
        off_z = distance * np.cos(zenith_rad)
        offset = Gf.Vec3d(off_x, off_y, off_z)

        # 2. 计算灯光在世界坐标系中的绝对位置
        # 绝对位置 = 中心点 + 极坐标偏移
        abs_pos = target_center + offset

        # 3. 获取 Xformable 接口
        xformable = UsdGeom.Xformable(prim)

        # 4. 设置位置 (Translate)
        translate_op = xformable.GetTranslateOp()
        if not translate_op:
            translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)

        # 这里的精度由 set_xform_op_smart 处理更稳妥，或者直接传 Vec3f
        # 考虑到位置属性通常也是 float，这里强制转换一下
        translate_op.Set(Gf.Vec3f(abs_pos))

        # 5. 计算朝向 (Orient) - 让灯光指向指定的 light_center
        # 方向向量 = 目标中心 - 灯光当前位置
        forward_dir = (target_center - abs_pos).GetNormalized()

        # 默认灯光轴向是 -Z (0,0,-1)
        default_dir = Gf.Vec3d(0, 0, -1)

        # 计算从默认轴到目标方向的旋转
        rot_matrix_quatd = Gf.Rotation(default_dir, forward_dir).GetQuat()

        # 6. 获取或创建 Orient 操作
        orient_op = xformable.GetOrientOp()
        if not orient_op:
            orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

        # set_xform_op_smart 内部会根据 orient_op 的实际类型进行转换数据精度Gf.Vec3d/Gf.Vec3f
        IlluminationUtils.set_xform_op_smart(orient_op, rot_matrix_quatd)

    @staticmethod
    def draw_illumination_visualization(zenith_deg, azimuth_deg, light_center=(0.0, 0.0, 0.0), line_length=200.0):
        """
        根据天顶角和方位角，围绕指定中心点绘制光源方向可视化线段

        Args:
            zenith_deg: 天顶角 (0-90, 0是正上方)
            azimuth_deg: 方位角 (0-360)
            light_center: 线段的中心坐标 (x, y, z), 默认 (0,0,0)
            line_length: 线段总长度 (默认50米)
        """
        # 1. 获取接口
        draw = _debug_draw.acquire_debug_draw_interface()

        # 如果需要每帧刷新，取消下面两行的注释以避免“残影”
        draw.clear_lines() 
        draw.clear_points()

        # 2. 角度转弧度
        zenith_rad = np.radians(zenith_deg)
        azimuth_rad = np.radians(azimuth_deg)

        # 3. 计算指向太阳的单位方向向量 (基于 Z 向上坐标系)
        dx = np.sin(zenith_rad) * np.cos(azimuth_rad)
        dy = np.sin(zenith_rad) * np.sin(azimuth_rad)
        dz = np.cos(zenith_rad)
        sun_dir = np.array([dx, dy, dz])

        # 4. 计算中心偏移
        center = np.array(light_center)
        half_len = line_length / 2.0

        # 线段两端：起点指向太阳，终点背向太阳
        start_point = center + (sun_dir * half_len)
        end_point = center - (sun_dir * half_len)

        # 5. 准备绘制参数
        # 转换为 float 元组列表
        p_starts = [(float(start_point[0]), float(start_point[1]), float(start_point[2]))]
        p_ends = [(float(end_point[0]), float(end_point[1]), float(end_point[2]))]

        # 颜色设置
        line_colors = [(1.0, 1.0, 0.0, 1.0)]  # 黄色线段
        center_color = [(0.0, 1.0, 1.0, 1.0)] # 青色中心点 (方便定位中心)
        point_colors = [(1.0, 0.5, 0.0, 1.0)] # 橙色端点 (指向太阳的一端)

        # 6. 执行绘制
        # 绘制主方向线
        draw.draw_lines(p_starts, p_ends, line_colors, [2.0])

        # 绘制中心点（可选，帮助确认中心位置）
        draw.draw_points([(float(center[0]), float(center[1]), float(center[2]))], center_color, [5.0])

        # 绘制指向太阳的端点（橙色大点）
        draw.draw_points(p_starts, point_colors, [12.0])

    @staticmethod
    def set_xform_op_smart(op, value):
        """
        智能根据 op 的精度设置数值
        """
        if not op:
            return

        target_type = op.GetAttr().GetTypeName()

        # 如果目标是单精度
        if "f" in str(target_type).lower():
            if isinstance(value, (Gf.Vec3d, Gf.Vec3f)):
                op.Set(Gf.Vec3f(value))
            elif isinstance(value, (Gf.Quatd, Gf.Quatf)):
                op.Set(Gf.Quatf(value))
        # 如果目标是双精度
        else:
            if isinstance(value, (Gf.Vec3d, Gf.Vec3f)):
                op.Set(Gf.Vec3d(value))
            elif isinstance(value, (Gf.Quatd, Gf.Quatf)):
                op.Set(Gf.Quatd(value))

    @staticmethod
    def get_scene_bounding_info(root_path="/World"):
        """
        计算场景中指定路径下所有几何体的包围盒、中心点和外接圆半径

        Args:
            root_path: 计算的起始路径，默认为 "/World"

        Returns:
            dict: 包含 min, max, center, radius, size 的字典
        """
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(root_path)
        # ----------------- 第一步：检查并删除"/World"下的光源 ------------
        # "/World"下的有边界的光源(disklight,clyberlight)会影响世界包围盒计算ComputeWorldBound
        # 注意：在遍历时删除元素，建议先转成 list 获取路径，再统一删除
        prims_to_delete = []

        # 遍历根路径下的直接子节点
        for child in root_prim.GetChildren():
            # 检查是否是灯光 (基于 API 或 类型名)
            if child.HasAPI(UsdLux.LightAPI) or "Light" in child.GetTypeName():
                prims_to_delete.append(child.GetPath())

        if prims_to_delete:
            for p_path in prims_to_delete:
                print(f"Cleaning up old light: {p_path}")
                stage.RemovePrim(p_path)
            # 提示：删除 Prim 后，Stage 会自动更新
        if not root_prim.IsValid():
            print(f"Warning: Root path {root_path} is invalid.")
            return None

        # ----------------- 第二步：计算几何包围盒 ------------
        # UsdGeom.Tokens.default_ 表示只计算常规几何体（不含 Proxy 或 Guide）
        purposes = [UsdGeom.Tokens.default_]
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)

        # 2. 计算世界坐标系下的包围盒
        # ComputeWorldBound 返回的是 Gf.BBox3d，包含了变换矩阵
        bbox = bbox_cache.ComputeWorldBound(root_prim)

        # 提取 AABB 范围 (Gf.Range3d)
        range_3d = bbox.ComputeAlignedRange()
        min_pt = range_3d.GetMin()
        max_pt = range_3d.GetMax()

        # 3. 如果场景为空，处理异常
        if range_3d.IsEmpty():
            return {"center": (0, 0, 0), "radius": 0, "size": (0, 0, 0)}

        # 4. 计算几何中心
        center = (min_pt + max_pt) / 2.0

        # 5. 计算包围盒尺寸 (长、宽、高)
        size = max_pt - min_pt

        # 6. 计算外接圆/球半径 (从中心到最远顶点的距离)
        # 这是最稳妥的外切圆计算方式
        radius = (max_pt - center).GetLength()

        return {
            "min": (min_pt[0], min_pt[1], min_pt[2]),
            "max": (max_pt[0], max_pt[1], max_pt[2]),
            "center": (center[0], center[1], center[2]),
            "size": (size[0], size[1], size[2]),
            "radius": float(radius)
        }


def solar_position(lat, lon, dt_utc):
    """
    给定经纬度(float)和日期(year, month, day, hour, minute, second)计算太阳位置altitude(deg), azimuth(deg)

    Parameters:
        lat/lon (float) : Latitude/Longitude in degrees;
        dt_utc (tuple) : UTC datetime (year, month, day, hour, minute, second).

    return:
        altitude(deg), azimuth(deg)
    """

    # ---------- 1. 儒略日 ----------
    timestamp = dt_utc.timestamp()
    jd = timestamp / 86400.0 + 2440587.5
    n = jd - 2451545.0

    # ---------- 2. 太阳黄经 ----------
    L = (280.46 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)

    lambda_sun = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))

    # ---------- 3. 赤纬 ----------
    epsilon = math.radians(23.439)
    delta = math.asin(math.sin(epsilon) * math.sin(lambda_sun))

    # ---------- 4. 时角 ----------
    time_utc_hours = (
        dt_utc.hour +
        dt_utc.minute / 60 +
        dt_utc.second / 3600)
    # 以地方真太阳时为基准
    solar_time = time_utc_hours + lon / 15.0
    H = math.radians((solar_time - 12.0) * 15.0)

    # ---------- 5. 高度角 ----------
    lat_rad = math.radians(lat)
    altitude = math.asin(
        math.sin(lat_rad) * math.sin(delta) +
        math.cos(lat_rad) * math.cos(delta) * math.cos(H))

    # ---------- 6. 方位角 ----------
    azimuth = math.atan2(
        -math.sin(H),
        math.cos(H) * math.sin(lat_rad) - math.tan(delta) * math.cos(lat_rad))
    azimuth = (math.degrees(azimuth) + 360) % 360

    return math.degrees(altitude), azimuth


def sun_positions_in_day(lat, lon, date, sunrise_step_minutes=60, simulation_hour_step=1):
    """
    给定经纬度和日期 (year, month, day),计算一天内的日出日落间的太阳位置
    已知问题: 最开始的sunrise时间太阳高度角接近0,辐射定标反射板没接收到光照

    Parameters:
        lat(float): Latitude in degrees;
        lon(float): Longitude in degrees;
        date(tuple(int, int, int)): Date as (year, month, day), UTC;
        sunrise_step_minutes(int): Time step (minutes) for sunrise/sunset detection.;
        simulation_hour_step(int): Time step (hours) for simulation sampling;
    return:
        List[dict]，每个元素包含：
            {
                "time": UTC datetime,
                "altitude": float (deg),
                "azimuth": float (deg)
            }
    """

    # 计算一天中所有的时刻和太阳位置
    dt = datetime(*date, 0, 0, tzinfo=timezone.utc)
    step = timedelta(minutes=sunrise_step_minutes)  # 间隔步长

    sun_table = []  # 每个元素: {"time", "alt", "azi"}

    for _ in range(int(24 * 60 / sunrise_step_minutes)):
        alt, azi = solar_position(lat, lon, dt)
        sun_table.append({
            "time": dt,
            "altitude": alt,
            "azimuth": azi
        })
        dt += step

    # 判定日出 / 日落的时间
    sunrise = None
    sunset = None
    for i in range(1, len(sun_table)):
        alt_prev = sun_table[i - 1]["altitude"]
        alt_curr = sun_table[i]["altitude"]

        if alt_prev < 0 and alt_curr >= 0:
            sunrise = sun_table[i]["time"]

        if alt_prev >= 0 and alt_curr < 0:
            sunset = sun_table[i]["time"]

    if sunrise is None or sunset is None:
        raise RuntimeError("Sunrise or sunset not found for given date/location.")

    # 只保留日出日落间的时间和太阳位置
    t = sunrise.replace(minute=0, second=0, microsecond=0)
    simulation_results = []
    while t <= sunset:
        # 在 sun_table 中找最接近 t 的太阳状态
        closest = min(
            sun_table,
            key=lambda s: abs((s["time"] - t).total_seconds())
        )

        simulation_results.append({
            "time": t,
            "altitude": closest["altitude"],
            "azimuth": closest["azimuth"]
        })

        t += timedelta(hours=simulation_hour_step)

    return simulation_results


def lst_to_utc(year, month, day, lst_hour, lst_minute, lon):
    '''卫星过境时间(当地太阳时)转世界标准时间(UTC时间)'''
    lst = lst_hour + lst_minute / 60.0
    utc_hours = lst - lon / 15.0

    hour = int(utc_hours)
    minute = int((utc_hours - hour) * 60)
    second = int((((utc_hours - hour) * 60) - minute) * 60)

    return datetime(
        year, month, day,
        hour % 24, minute, second,
        tzinfo=timezone.utc
    )


def sun_positions_in_year(lat, lon, year, lst_hour=10, lst_minute=30, step_days=20):
    """
    计算一年内，每隔 step_days 天，在 10:30 (UTC) 的太阳高度角和方位角

    Parameters:
        lat (float)       : Latitude (deg)
        lon (float)       : Longitude (deg)
        year (int)        : Year, e.g. 2024
        step_days (int)   : Interval in days

    Returns:
        List[dict], each element contains:
        {
            "time": UTC datetime,
            "altitude": float (deg),
            "azimuth": float (deg)
        }
    """

    results = []

    # 一年起始时间：1 月 1 日 10:30 UTC
    # dt = datetime(year, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    dt = lst_to_utc(year=year, month=1, day=1, lst_hour=lst_hour, lst_minute=lst_minute, lon=lon)  # 当地太阳时转UTC时间
    end_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    while dt < end_dt:
        altitude, azimuth = solar_position(lat, lon, dt)

        results.append({
            "time": dt,
            "altitude": altitude,
            "azimuth": azimuth
        })

        dt += timedelta(days=step_days)

    return results


# 测试数据
# results = sun_positions_in_year(lat=29.13, lon=110.47, year=2025, lst_hour=10, lst_minute=30, step_days=20)
# print(results)
