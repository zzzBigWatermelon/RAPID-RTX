__all__ = ["ObservationWindow"]


from pxr import UsdGeom, Sdf
import omni.kit.commands
import asyncio
import omni.ui as ui
import omni.usd
import carb
from pathlib import Path
import os
from isaacsim.gui.components.style import get_style
# 自定义功能
from rapid.Utility.window_components_combo_box_model import ComboBoxModel
from .observation_utils import ObservationUtils
from rapid.Utility.custom_json_encoder import CompactListEncoder  # json格式编码器
from rapid.Utility.searchable_prim_picker import SearchablePrimPicker  # 舞台prim搜索选择功能
from .create_update_optics_sensor import create_optics_sensor
from .create_update_LiDAR_sensor import create_LiDAR_carrier


# 定义文件路径
DATA_ROOT = Path(__file__).parent.parent.parent/'data'
LABEL_WIDTH = 120
SPACING = 8


class ObservationWindow(ui.Window):
    """这个类设计窗口的具体层次结构"""

    def __init__(self, title: str, delegate=None, **kwargs):
        super().__init__(title, **kwargs)
        self.__label_width = LABEL_WIDTH

        # 1. 所有的 UI Model 数据持久化 (UI 清空时数据不会丢)
        self._init_models()

        # 监听新打开Stage事件
        self._usd_context = omni.usd.get_context()
        # 订阅事件流 (OPENED, SAVED, CLOSED 等)
        self._stage_event_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
           self._on_stage_event
        )

        # Apply the style to all the widgets of this window
        # self.frame.style = scatter_window_style
        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_main_obversevation_ui)

    @property
    def label_width(self):
        """The width of the attribute label"""
        return self.__label_width

    @label_width.setter
    def label_width(self, value):
        """调用label_width并重新赋值时,会重新刷新
        """
        self.__label_width = value
        # 用于通知框架或界面重新构建或刷新，以反映属性值的变化。
        self.frame.rebuild()

    def _init_models(self):
        # 初始化专门存储模型的字典
        # UI模型直接存入字典，键名与最终输出数据的键名保持一致
        self.models = {}

        # 通用属性
        self.models["sensor_stage_path"] = ui.SimpleStringModel('/World/sensor')
        self.models["sensor_type"] = ComboBoxModel("Perspective", "Orthographic", "Airborne LiDAR", "Terrestrial LiDAR", "Spaceborne LiDAR (Waveform LiDAR Data)")
        self.models["optical_observation_type"] = ComboBoxModel("Single Sampling", "Constant Altitude Sampling",
                                                                "Semi-circular Sampling", "Omnidirectional Sampling")

        # 透视传感器属性
        self.models["optical_sensor_pixels"] = [ui.SimpleIntModel(v) for v in (200, 200)]  # 光学分辨率
        self.models["perspective_sensor_fov"] = ui.SimpleFloatModel(30.0)
        # 正射传感器属性
        self.models["orthographic_sensor_extent"] = ui.SimpleFloatModel(30.0)

        # 光学传感器恒定单次采样
        self.models["single_sampling_sensor_position"] = [ui.SimpleFloatModel(v) for v in (0, 0, 30)]
        self.models["single_sampling_observation_center"] = [ui.SimpleFloatModel(v) for v in (0, 0, 0)]
        # 光学传感器恒定高度飞行采样
        self.models["constant_altitude_sampling_start_point"] = [ui.SimpleFloatModel(v) for v in (0, 0)]
        self.models["constant_altitude_sampling_end_point"] = [ui.SimpleFloatModel(v) for v in (100, 100)]
        self.models["constant_altitude_sampling_flight_altitude"] = ui.SimpleFloatModel(30.0)
        self.models["constant_altitude_sampling_flight_speed"] = ui.SimpleFloatModel(10.0)
        self.models["constant_altitude_sampling_forward_and_side_overlap"] = [ui.SimpleFloatModel(v) for v in (50, 50)]
        # 光学传感器竖直半圆飞行采样
        self.models["semicircular_sampling_observation_center"] = [ui.SimpleFloatModel(v) for v in (0, 0, 0)]
        self.models["semicircular_sampling_distance"] = ui.SimpleFloatModel(30.0)
        self.models["semicircular_sampling_view_azimuth"] = ui.SimpleFloatModel(0.0)
        self.models["semicircular_sampling_zenith_range"] = [ui.SimpleFloatModel(v) for v in (60, -60)]
        self.models["semicircular_sampling_zenith_step"] = ui.SimpleFloatModel(5.0)
        # 光学传感器多水平圆环采样
        self.models["omnidirectional_sampling_observation_center"] = [ui.SimpleFloatModel(v) for v in (0, 0, 0)]
        self.models["omnidirectional_sampling_distance"] = ui.SimpleFloatModel(30.0)
        self.models["omnidirectional_sampling_view_zenith"] = ui.SimpleStringModel('20.0,30.0,40.0')
        self.models["omnidirectional_sampling_azimuth_range"] = [ui.SimpleFloatModel(v) for v in (60, -60)]
        self.models["omnidirectional_sampling_azimuth_step"] = ui.SimpleFloatModel(5.0)

        # 机载激光雷达
        self.models["airborne_LiDAR_fov"] = ui.SimpleFloatModel(80.0)
        self.models["airborne_LiDAR_angle_resolution"] = ui.SimpleFloatModel(0.05)
        self.models["airborne_LiDAR_scan_rate"] = ui.SimpleFloatModel(60)
        self.models["airborne_LiDAR_start_point"] = [ui.SimpleFloatModel(v) for v in (0, 0)]
        self.models["airborne_LiDAR_end_point"] = [ui.SimpleFloatModel(v) for v in (100, 100)]
        self.models["airborne_LiDAR_flight_altitude"] = ui.SimpleFloatModel(60)
        self.models["airborne_LiDAR_flight_speed"] = ui.SimpleFloatModel(10)
        self.models["airborne_LiDAR_strip_overlap"] = ui.SimpleFloatModel(40)

        # 地基激光雷达
        self.models["terrestrial_LiDAR_min_zenith_angle"] = ui.SimpleFloatModel(30.0)
        self.models["terrestrial_LiDAR_max_zenith_angle"] = ui.SimpleFloatModel(150.0)
        self.models["terrestrial_LiDAR_zenith_angle_resolution"] = ui.SimpleFloatModel(0.1)
        self.models["terrestrial_LiDAR_azimuth_angle_resolution"] = ui.SimpleFloatModel(0.1)
        self.models["terrestrial_LiDAR_sampling_frequency"] = ui.SimpleFloatModel(100000)
        self.models["terrestrial_LiDAR_position"] = [ui.SimpleFloatModel(v) for v in (0, 0, 5)]

        # 星载激光雷达波形模拟
        self.models["spaceborne_LiDAR_footprint_width"] = ui.SimpleFloatModel(20.0)  # 采用的是4倍高斯标准差4σ_f
        self.models["spaceborne_LiDAR_system_pulse_width"] = ui.SimpleFloatModel(0.8)
        self.models["spaceborne_LiDAR_vertical_bin_size"] = ui.SimpleFloatModel(0.15)
        self.models["spaceborne_LiDAR_footprint_center"] = [ui.SimpleFloatModel(v) for v in (0, 0)]

        # ... 其他所有模型照样填入

    def _on_stage_event(self, event: carb.events.IEvent):
        # 检查事件类型是否为 "Stage 已打开"
        if event.type == int(omni.usd.StageEventType.OPENED):
            carb.log_info("Detected Stage Opened event. Syncing UI parameters from JSON...")

            # 确保 models 已经初始化
            if hasattr(self, "models") and self.models:
                ObservationUtils.read_json_data_to_UI(self.models)
            else:
                carb.log_warn("UI models not initialized yet, skipping sync.")

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def _build_main_obversevation_ui(self):
        """
        一级结构：主滚动窗口，包含所有大组
        """
        with ui.ScrollingFrame():
            with ui.VStack(spacing=5, height=0):
                # 1. 第一大组：Sensor（包含设置和参数）
                self._build_sensor_params_group()

                # 2. 第二大组：Observation Geometry（包含观测模式）
                self._build_observation_modes_group()

                # 3. 底部固定按钮
                ui.Spacer(height=10)
                ui.Button("Create Sensor", height=40, clicked_fn=self._on_create_clicked)
                ui.Spacer(height=20)

        # 初次运行，手动触发一次全量构建
        self._refresh_dynamic_ui()

    def _build_sensor_params_group(self):
        """对应 groupFrame: Sensor params"""
        with ui.CollapsableFrame(title="Sensor Params", name="groupFrame", height=0, style=get_style()):
            with ui.VStack(height=0, spacing=SPACING):
                # 子组 1: Sensor Settings (固定内容：路径选择和类型切换)
                with ui.CollapsableFrame(title="Sensor Settings", name="subFrame", height=0, style=get_style()):
                    with ui.VStack(spacing=SPACING):
                        with ui.HStack(height=0):
                            ui.Label("Stage Sensor Prim:", width=self.__label_width)
                            ui.StringField(model=self.models["sensor_stage_path"])
                            search_icon_path = os.path.join(DATA_ROOT, "search.svg")
                            ui.Button(width=30, height=22, image_url=search_icon_path, clicked_fn=self._on_open_picker_clicked)

                        with ui.HStack(height=0):
                            ui.Label("Sensor Type:", width=self.__label_width)
                            # 绑定回调：类型切换时刷新下方所有容器
                            ui.ComboBox(self.models["sensor_type"]).model.add_item_changed_fn(
                                lambda m, i: self._refresh_dynamic_ui()
                            )

                # 子组 2: Sensor Parameters (动态内容：随类型改变)
                with ui.CollapsableFrame(title="Sensor Parameters", name="subFrame", height=0, style=get_style()):
                    # 这里是动态占位符
                    self._params_container = ui.Frame()

    def _build_observation_modes_group(self):
        """对应 groupFrame: Observation Geometry"""
        with ui.CollapsableFrame(title="Observation Geometry", name="groupFrame", height=0, style=get_style()):
            with ui.VStack(height=0, spacing=SPACING):
                # 子组: Observation Mode (动态内容)
                with ui.CollapsableFrame(title="Observation Mode",
                                         name="subFrame",
                                         height=0,
                                         style=get_style(),
                                         horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                         vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    # 这里是动态占位符
                    self._modes_container = ui.Frame()

    def _refresh_dynamic_ui(self):
        """Sensor Parameters和Observation Mode都是随着Sensor type而改变的
        清空并重新填充所有动态区域"""
        sensor_index = self.models["sensor_type"].get_item_value_model(None, 0).as_int

        # 1. 刷新传感器参数区
        self._params_container.clear()
        with self._params_container:
            if sensor_index == 0:
                self._build_perspective_sensor()
            elif sensor_index == 1:
                self._build_orthographic_sensor()
            elif sensor_index == 2:
                self._build_airborne_lidar_sensor()
            elif sensor_index == 3:
                self._build_terrestrial_lidar_sensor()
            elif sensor_index == 4:
                self._build_spaceborne_lidar_sensor()

        # 2. 刷新观测模式区
        self._modes_container.clear()
        with self._modes_container:
            if sensor_index in [0, 1]:  # 光学相机
                self._build_optical_observation()
            elif sensor_index == 2:    # 机载雷达
                self._build_airborne_lidar_observation()
            elif sensor_index == 3:    # 地基雷达
                self._build_terrestrial_lidar_observation()
            elif sensor_index == 4:
                self._build_spaceborne_lidar_observation()

    def _build_perspective_sensor(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Width [px]:", width=self.__label_width)
                ui.IntField(model=self.models["optical_sensor_pixels"][0])
                ui.Spacer(width=10)
                ui.Label("Height [px]:", width=self.__label_width)
                ui.IntField(model=self.models["optical_sensor_pixels"][1])
            with ui.HStack():
                ui.Label("FOV [°]:", width=self.__label_width)
                ui.FloatField(model=self.models["perspective_sensor_fov"])

    def _build_orthographic_sensor(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Width [pixels]:", name="Width [pixels]", width=self.label_width, style=get_style())
                ui.IntField(model=self.models["optical_sensor_pixels"][0])
                ui.Label(" ", width=40)  # 在Width [pixels]和Height [Pixels]之间增加间距
                ui.Label("Height [Pixels]:", name="Height [Pixels]", width=self.label_width, style=get_style())
                ui.IntField(model=self.models["optical_sensor_pixels"][1])
            with ui.HStack():
                ui.Label("Diagonal Extent [m]:", name="Diagonal Extent", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["orthographic_sensor_extent"])

    def _build_airborne_lidar_sensor(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Field of View [°]:", name="Field of View [°]", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["airborne_LiDAR_fov"])
            with ui.HStack():
                ui.Label("Angle Resolution [°]:", name="Angle Resolution [°]", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["airborne_LiDAR_angle_resolution"])
            with ui.HStack():
                ui.Label("Scan Rate [Hz]:", name="Scan Rate", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["airborne_LiDAR_scan_rate"])

    def _build_terrestrial_lidar_sensor(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Min Zenith Angle [°]:", name="Min Zenith Angle", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["terrestrial_LiDAR_min_zenith_angle"])
                ui.Label("Max Zenith Angle [°]:", name="Max Zenith Angle", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["terrestrial_LiDAR_max_zenith_angle"])
            with ui.HStack():
                ui.Label("Zenith Angle Resolution [°]:", name="Zenith Angle Resolution", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["terrestrial_LiDAR_zenith_angle_resolution"])
                ui.Label("Azimuth Angle Resolution [°]:", name="Azimuth Angle Resolution", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["terrestrial_LiDAR_azimuth_angle_resolution"])
            with ui.HStack():
                ui.Label("Sampling Frequency [Hz]:", name="Zenith Angle Resolution", width=self.label_width, style=get_style())
                ui.IntField(model=self.models["terrestrial_LiDAR_sampling_frequency"])

    def _build_spaceborne_lidar_sensor(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Footprint Width [m]:", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["spaceborne_LiDAR_footprint_width"])
            with ui.HStack():
                ui.Label("System Pulse Width [m]:", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["spaceborne_LiDAR_system_pulse_width"])
            with ui.HStack():
                ui.Label("Vertical Bin Size [m]:", width=self.label_width, style=get_style())
                ui.FloatField(model=self.models["spaceborne_LiDAR_vertical_bin_size"])

    def _build_optical_observation(self):
        """光学模式包含二级动态刷新（采样模式切换）"""
        with ui.VStack(height=0, spacing=SPACING):
            with ui.HStack():
                ui.Label("Sampling Model:", width=self.__label_width, style=get_style())
                # 二级切换：采样模式改变时，只刷新模式区内部
                ui.ComboBox(self.models["optical_observation_type"]).model.add_item_changed_fn(
                    lambda m, i: self._build_refresh_optical_sampling_content()
                )

            # 模式内部的内容动态容器
            self._sampling_content_frame = ui.Frame()
            self._build_refresh_optical_sampling_content()

    def _build_refresh_optical_sampling_content(self):
        """专门刷新光学采样模式的具体参数"""
        sampling_index = self.models["optical_observation_type"].get_item_value_model(None, 0).as_int
        self._sampling_content_frame.clear()
        with self._sampling_content_frame:
            if sampling_index == 0:   # 单次采样
                with ui.VStack(spacing=SPACING):
                    with ui.HStack():
                        ui.Label("Camera Position:", name="Camera Position", width=self.label_width)
                        ui.Label("X [m]:", name="Camera Position X", width=self.label_width)
                        ui.FloatField(model=self.models["single_sampling_sensor_position"][0])
                        ui.Label("Y [m]:", name="Camera Position Y", width=self.label_width)
                        ui.FloatField(model=self.models["single_sampling_sensor_position"][1])
                        ui.Label("Z [m]:", name="Camera Position Z", width=self.label_width)
                        ui.FloatField(model=self.models["single_sampling_sensor_position"][2])
                    with ui.HStack():
                        ui.Label("Observation Center:", name="Sensor Height", width=self.label_width)
                        ui.Label("X [m]:", name="Observation Center X", width=self.label_width)
                        ui.FloatField(model=self.models["single_sampling_observation_center"][0])
                        ui.Label("Y [m]:", name="Observation Center Y", width=self.label_width)
                        ui.FloatField(model=self.models["single_sampling_observation_center"][1])
                        ui.Label("Z [m]:", name="Observation Center Z", width=self.label_width)
                        ui.FloatField(model=self.models["single_sampling_observation_center"][2])

            elif sampling_index == 1:  # Constant Altitude
                with ui.VStack(spacing=SPACING):
                    with ui.HStack():
                        ui.Label("Start Point:", name="Start Point", width=self.label_width)
                        ui.Label("X [m]:", name="Start Point X", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_start_point"][0])
                        ui.Label("Y [m]:", name="Start Point Y", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_start_point"][1])
                    with ui.HStack():
                        ui.Label("End Point:", name="End Point", width=self.label_width)
                        ui.Label("X [m]:", name="End Point X", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_end_point"][0])
                        ui.Label("Y [m]:", name="End Point Y", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_end_point"][1])
                    with ui.HStack():
                        ui.Label("Flight Altitude [m]:", name="Flight Altitude", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_flight_altitude"])
                        ui.Label("Flight Speed [m/s]:", name="Flight Altitude", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_flight_speed"])
                    with ui.HStack():
                        ui.Label("Forward Overlap Ratio (%):", name="Forward Overlap", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_forward_and_side_overlap"][0])
                        ui.Label("Side Overlap Ratio (%):", name="Side Overlap", width=self.label_width)
                        ui.FloatField(model=self.models["constant_altitude_sampling_forward_and_side_overlap"][1])

            elif sampling_index == 2:  # Semi-circular Sampling
                with ui.VStack(spacing=SPACING):
                    with ui.HStack():
                        ui.Label("Observation Center:", name="Start Point", width=self.label_width)
                        ui.Label("X [m]:", name="Start Point X", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_observation_center"][0])
                        ui.Label("Y [m]:", name="Start Point Y", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_observation_center"][1])
                        ui.Label("Z [m]:", name="Start Point Y", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_observation_center"][2])
                    with ui.HStack():
                        ui.Label("Distance [m]:", name="Distance", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_distance"])
                        ui.Label("View Azimuth [°]:", name="Distance", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_view_azimuth"])
                    with ui.HStack():
                        ui.Label("View Zenith Range:", name="View Zenith Range", width=self.label_width)
                        ui.Label("Start Angle [°]:", name="Start Angle", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_zenith_range"][0])
                        ui.Label("End Angle [°]:", name="End Angle", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_zenith_range"][1])
                        ui.Label("Zenith Step [°]:", name="Distance", width=self.label_width)
                        ui.FloatField(model=self.models["semicircular_sampling_zenith_step"])

            elif sampling_index == 3:  # omnidirectional sampling
                with ui.VStack(spacing=SPACING):
                    with ui.HStack():
                        ui.Label("Observation Center:", name="Start Point", width=self.label_width)
                        ui.Label("X [m]:", width=self.label_width)
                        ui.FloatField(model=self.models["omnidirectional_sampling_observation_center"][0])
                        ui.Label("Y [m]:", width=self.label_width)
                        ui.FloatField(model=self.models["omnidirectional_sampling_observation_center"][1])
                        ui.Label("Z [m]:", width=self.label_width)
                        ui.FloatField(model=self.models["omnidirectional_sampling_observation_center"][2])
                    with ui.HStack():
                        ui.Label("Distance [m]:", name="Distance", width=self.label_width)
                        ui.FloatField(model=self.models["omnidirectional_sampling_distance"])
                        ui.Label("Zenith [°]:", name="Zenith", width=self.label_width)
                        ui.StringField(model=self.models["omnidirectional_sampling_view_zenith"])
                        ui.Label("Zenith Step [°]:", name="Zenith Step", width=self.label_width)
                        ui.FloatField(model=self.models["omnidirectional_sampling_azimuth_step"])

    def _build_airborne_lidar_observation(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Start Point:", name="Start Point", width=self.label_width)
                ui.Label("X [m]:", name="Start Point X", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_start_point"][0])
                ui.Label("Y [m]:", name="Start Point Y", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_start_point"][1])
            with ui.HStack():
                ui.Label("End Point:", name="End Point", width=self.label_width)
                ui.Label("X [m]:", name="End Point X", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_end_point"][0])
                ui.Label("Y [m]:", name="End Point Y", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_end_point"][1])
            with ui.HStack():
                ui.Label("Flight Altitude [m]:", name="Flight Altitude", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_flight_altitude"])
                ui.Label("Flight Speed [m/s]:", name="Flight Altitude", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_flight_speed"])
            with ui.HStack():
                ui.Label("Forward Overlap Ratio (%):", name="Forward Overlap", width=self.label_width)
                ui.FloatField(model=self.models["airborne_LiDAR_strip_overlap"])

    def _build_terrestrial_lidar_observation(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Position:", name="Start Point", width=self.label_width)
                ui.Label("X [m]:", name="Start Point X", width=self.label_width)
                ui.FloatField(model=self.models["terrestrial_LiDAR_position"][0])
                ui.Label("Y [m]:", name="Start Point Y", width=self.label_width)
                ui.FloatField(model=self.models["terrestrial_LiDAR_position"][1])
                ui.Label("Z [m]:", name="Start Point Y", width=self.label_width)
                ui.FloatField(model=self.models["terrestrial_LiDAR_position"][2])

    def _build_spaceborne_lidar_observation(self):
        with ui.VStack(spacing=SPACING):
            with ui.HStack():
                ui.Label("Footprint Center:", name="Start Point", width=self.label_width)
                ui.Label("X [m]:", name="Start Point X", width=self.label_width)
                ui.FloatField(model=self.models["spaceborne_LiDAR_footprint_center"][0])
                ui.Label("Y [m]:", name="Start Point Y", width=self.label_width)
                ui.FloatField(model=self.models["spaceborne_LiDAR_footprint_center"][1])

    def _on_open_picker_clicked(self):
        """点击搜索图标按钮时的回调：弹出选择器"""
        SearchablePrimPicker(
            title="Pick a Camera Prim",
            type_filter=UsdGeom.Camera,  # 过滤只看相机
            on_select_fn=self._on_prim_selected_callback  # 选中后的回调
        )

    def _on_prim_selected_callback(self, path):
        """当用户在弹窗里点击了某个路径时"""
        # 将选中的路径更新到你定义的 SimpleStringModel 中
        self.models["sensor_stage_path"].as_string = path
        print(f"ObservationWindow 收到新路径: {path}")

    def _on_create_clicked(self):
        # 自动遍历全部模型并提取值
        data = {}
        for key, model in self.models.items():
            raw_value = ObservationUtils.get_ui_value(model)  # 调用提取函数
            if key == "omnidirectional_sampling_view_zenith":
                # 从str类型中拆分出float类型数据
                data[key] = [float(x.strip()) for x in raw_value.split(',') if x.strip()]
            else:
                data[key] = raw_value

        # 先删除之前的路径下的prim
        stage = omni.usd.get_context().get_stage()
        sensor_path = self.models["sensor_stage_path"].as_string
        prim = stage.GetPrimAtPath(sensor_path)
        if prim.IsValid():
            omni.kit.commands.execute('DeletePrims',
                paths=[Sdf.Path(sensor_path)],
                destructive=False)

        # 创建光学传感器，sensor类型的prim，可以直接使用
        sensor_type = data.get("sensor_type")
        if sensor_type in ["Perspective", "Orthographic"]:
            create_optics_sensor(data)
        # 创建光学传感器，xform类型的prim，直记录传递数值
        elif sensor_type in ["Airborne LiDAR", "Terrestrial LiDAR", "Spaceborne LiDAR (Waveform LiDAR Data)"]:
            asyncio.ensure_future(create_LiDAR_carrier(data))

        # 保存当前窗口文件至simulation_parameters.json文件
        ObservationUtils.save_UI_data_to_json(self.models)

    def get_data(self):
        return {
            "sensor_stage_path": self.models["sensor_stage_path"].as_string
        }
