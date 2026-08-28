__all__ = ["Simulation"]
# replicator模块
import omni.replicator.core as rep
from .WorkWriter import WorkWriter
# omni模块
import carb.settings
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
import omni
import omni.kit.viewport.utility as vp_utils
import omni.usd
import omni.kit.app
import omni.kit.actions.core
import omni.kit.commands
import omni.kit.notification_manager as nm
from isaacsim.util.debug_draw import _debug_draw
# 常用模块
import time
import os
import asyncio
from pathlib import Path
import shutil
import math
from typing import Dict
# 自定义模块
from rapid.Utility import project_validity_check  # 项目有效性检查
from rapid.Utility.illumination_utils import IlluminationUtils  # 计算当地太阳的位置
from rapid.Utility.sensor_params import get_prim_attributes  # 不同观测模式对应的属性
from rapid.Utility.calculate_sampling_waypoints import calculate_constant_altitude_sampling_waypoints  # 计算航线位置
from rapid.Utility.calculate_sampling_waypoints import calculate_semi_circular_sampling_waypoints  # 计算航线位置
from rapid.Utility.calculate_sampling_waypoints import calculate_omnidirectional_sampling_waypoints
from rapid.Utility.simulation_progress_window import SimulationProgressWindow  # 进度条窗口
from .simulation_utils import MaterialUtils, CameraUtils  # 材质反射率改变
from .radiometric_calibration import RadiometricCalibration  # 辐射定标计算
from .data_post_processing import NpyToReflectanceHdrConverter, BRFCurveCalculation  # npy缓存转.hdr反射率文件


# 反射板资产位置,反射率为0.03,0.05,0.07,语义属性Semantic Typere:reflectance_panel,Semantic Data:reflectance_panel_3/5/7
REFLECTANCE_PANEL_ASSET = str(Path(__file__).parent.parent.parent/'data'/'reflectance_panel_isaac4.5.usd')
# 反射板信息
REFLECTANCE_VALUES = [0.03, 0.05, 0.07]
REFLECTANCE_KEYS = ['3%', '5%', '7%']
REFLECTANCE_SEMANTIC_TYPE = 'reflectance_panel'
REFLECTANCE_SEMANTIC_DATA = ['reflectance_panel_3', 'reflectance_panel_5', 'reflectance_panel_7']


class Simulation:
    """"""

    def __init__(self):
        # 在初始化时定义观测反射板时需显示的系统白名单
        self._system_paths = {
            "/Replicator", "/Render", "/OmniverseKit_Persp",
            "/OmniverseKit_Front", "/OmniverseKit_Top",
            "/OmniverseKit_Right", "/Session"
        }
        self._cached_scene_prims = []
        self._cached_light_prims = []

        # 用于保存进度条窗口引用
        self._progress_win = None

    def start(self, simulation_parameters: Dict = {}) -> None:
        '''开始整个仿真流程

        参数:
        simulation_parameters (Dict): 仿真参数

        返回:
        '''
        # --------------------------------开始初始化-----------------------
        # 用于标记是否处于GUI模式，防止在 Headless 模式下运行 UI 代码导致报错
        self._is_gui_mode = True
        # 获取数据输出文件夹的路径
        self.intermediate_path = project_validity_check.get_folder("intermediate_data")
        self.parameters_path = project_validity_check.get_folder("parameters")
        self.result_path = project_validity_check.get_folder("result")
        # 删除上一次辐射定标的所有中间文件
        shutil.rmtree(self.intermediate_path, ignore_errors=True)
        # 重新创建空文件夹
        os.makedirs(self.intermediate_path, exist_ok=True)
        # stage/carb接口
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self._settings = carb.settings.get_settings()

        # --------------------------------参数计算-----------------------
        # 解析观测/场景几何参数
        parameters_ready = self.parse_simulation_parameters(simulation_parameters)
        # 解析反射率参数
        self.check_band_length_consistency(simulation_parameters)
        # 没有创建传感器不执行后面的模拟
        if not parameters_ready:
            return

        # 清除可视化航线
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_points()
        draw.clear_lines()

        # 开始模拟
        asyncio.ensure_future(self._run_simulation())

    async def _run_simulation(self):
        '''异步执行的外部包装函数
        '''

        try:
            # 进入模拟前,进行模拟环境配置设定
            self.prepare_simulation_environment()

            # 执行模拟
            start_time = time.time()  # 记录开始时间
            await self.__simulation_core()
        except Exception as e:
            if self._progress_win:
                self._progress_win.status_label.text = f"Error: {str(e)}"
            raise
        finally:
            # --------模拟结束后释放进程/显存与后处理---------
            # 结束模拟，释放进程/显存
            self.writer.detach()
            self.render_product.destroy()
            # Wait for the data to be written to disk
            await rep.orchestrator.wait_until_complete_async()

            # 退出模拟后,恢复可视化环境设定
            await self.cleanup_simulation_environment(start_time)
            # 后处理，文件移动，分类，转换和辐射定标等,BRF曲线计算
            self.__radiometric_calibration(self.bands_data)

    async def __simulation_core(self):
        '''使用replicator的核心模拟步骤'''
        # 观测位置设定
        observation_position, target_position = self.__calculate_observation_position()
        print('11111111111111111111111111111111111111111111111111111')
        print(observation_position)
        print('11111111111111111111111111111111111111111111111111111')

        # 计算光源位置，创建新光照
        self.create_and_setup_light()

        # 修改反射率数值事件，返回单个位置的模拟次数
        bands_per_position_number = MaterialUtils.updata_stage_materials(self.reflectance_data)

        # Set the renderer to Path Traced,
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
        pixels = self.params['optical_sensor_pixels']
        self.render_product = rep.create.render_product(self.sensor_path, resolution=(pixels[0], pixels[1]))

        # 初始化写入器和链接渲染产品
        self.writer = rep.WriterRegistry.get("WorkWriter")
        self.writer.initialize(
            output_dir=self.intermediate_path,
            rgb=True,
            hdr=True,
            bounding_box_2d_tight=True,
            semantic_types=['reflectance_panel'])
        self.writer.attach([self.render_product])

        # 场景中引入反射板
        self._progress_win.status_label.text = "Radiometric Calibration"  # 窗口上的内容
        calib_camera_pos, calib_target_pos, reflectance_panel = self.__add_reflectance_panel(panel_size=3, fill_factor=0.2)
        # 开始辐射定标观测
        await self.observe_reflectance_panel(reflectance_panel, calib_camera_pos, self.writer)

        # 计算模拟次数
        observation_position_number = len(observation_position)
        # observation_position_number = 2  # 测试
        # 进度条窗口总数计算
        current_step = 0
        self._progress_win.status_label.text = "Run Simulation..."  # 窗口上的内容
        self._progress_win.total_steps = observation_position_number*bands_per_position_number

        # 开始场景步进模拟
        # 传感器位置变换循环
        for i in range(observation_position_number):
            # 改变传感器位置
            self.__sensor_motion_control(observation_position, target_position, i)
            # self.draw.draw_points([observation_position[i]], [(1, 0, 0, 1)], [25])  # 可视化

            # 波段变换循环
            for j in range(bands_per_position_number):
                # 生成自定义前缀：pos001__band0001
                # i+1 和 j+1 是为了让文件名从 1 开始，03d 表示固定 3 位宽度，不足补零，(j+1)*3一个图片3个波段
                custom_prefix = f"pos{i+1:04d}_band{(j+1)*3:04d}"
                self.writer.set_custom_name(custom_prefix)
                # 改变物体反射率波段
                rep.utils.send_og_event(event_name="change_color")
                # 开始单次异步模拟
                await rep.orchestrator.step_async()
                # 更新进度条窗口
                current_step += 1
                self._progress_win.update_progress(current_step, custom_prefix)

    def parse_simulation_parameters(self, simulation_parameters: Dict = {}):
        '''解析读取仿真参数
        参数:
            simulation_complete (dict):数据格式如下{
            'Observation': {'sensor_stage_path': '/World/sensor'}
            'ReflectanceDatabase': {'leaf': {'ref': [0.2, 0.3, 0.4,...], 'tra': [0.2, 0.3, 0.4,...]}, 'Name': {'ref': [0.2, 0.3, 0.4,...], 'tra': [0.2, 0.3, 0.4,...]}}
            }
        '''
        # ----------------------通过事件流数据获取对应的传感器--------------------------
        # 获取舞台和传感器prim
        stage = omni.usd.get_context().get_stage()
        observation_data = simulation_parameters['Observation']
        self.sensor_path = observation_data['sensor_stage_path']
        self.sensor_prim = stage.GetPrimAtPath(self.sensor_path)
        if not self.sensor_prim.IsValid():
            # 如果不存在则直接返回
            nm.post_notification(
                "Sensor is not created. Please create a sensor before starting simulation.",
                status=nm.NotificationStatus.WARNING,
                duration=5)
            return False

        # ----------------------通过参数配置表获取传感器参数--------------------------
        self.params = {}
        # 首先读取基础参数 (Base_Sensor)
        get_prim_attributes(self.sensor_prim, "Base Attribute", self.params)
        # 读取光学传感器参数
        sensor_type = self.params.get("sensor_type")
        if sensor_type:
            get_prim_attributes(self.sensor_prim, sensor_type, self.params)
        # 读取观测模式的参数
        obs_type = self.params.get("optical_observation_type")
        if obs_type:
            get_prim_attributes(self.sensor_prim, obs_type, self.params)

        # 获取传感器USD原生属性
        camera_geom = UsdGeom.Camera(self.sensor_prim)
        self._horizontal_aperture_attr = camera_geom.GetHorizontalApertureAttr()
        self._focal_length_value_attr = camera_geom.GetFocalLengthAttr()
        self.camera_translate_op = camera_geom.GetTranslateOp()

        # ----------------------获取太阳几何参数--------------------------
        scene_construction_data = simulation_parameters['SceneConstruction']
        self.params['light_zenith_and_azimuth'] = scene_construction_data['light_zenith_and_azimuth']
        self.params['direct_sun_intensity'] = scene_construction_data['direct_sun_intensity']
        self.params['diffuse_sky_intensity'] = scene_construction_data['diffuse_sky_intensity']
        return True

    def check_band_length_consistency(self, simulation_parameters):
        """
        检查所有 ref 和 tra 的波段长度是否一致,以最大长度为基准
        如果长度不足,使用最后一个值补全
        {'leaf': {'ref': [0.2, 0.3, 0.4,...], 'tra': [0.2, 0.3, 0.4,...], 'display_color': [0.1,0.1,0.1]}, 'Name': {'ref': [0.2, 0.3, 0.4,...], 'tra': [0.2, 0.3, 0.4,...]}}

        返回:
            max_len
        """
        self.reflectance_data = simulation_parameters['ReflectanceDatabase']['ref_tra_data']
        self.bands_data = simulation_parameters['ReflectanceDatabase']['bands_data']
        print('11111111111111111111111111111111111111111111111111111111')
        print(self.reflectance_data)
        print(self.bands_data)
        print('11111111111111111111111111111111111111111111111111111111')
        lengths = []
        # 1. 收集所有长度
        for name, data in self.reflectance_data.items():
            ref_len = len(data.get('ref', []))
            tra_len = len(data.get('tra', []))

            lengths.append(ref_len)
            lengths.append(tra_len)

        if not lengths:
            return 0

        max_len = max(lengths)
        mismatch_items = []

        # 2. 补全数据
        for name, data in self.reflectance_data.items():
            ref_list = data.get('ref', [])
            tra_list = data.get('tra', [])
            ref_len = len(ref_list)
            tra_len = len(tra_list)
            # ---------- 补全 ref ----------
            if ref_len > 0 and ref_len < max_len:
                last_val = ref_list[-1]
                ref_list.extend(
                    [last_val] * (max_len - ref_len))
                mismatch_items.append(
                    f"{name}.ref({ref_len}->{max_len})")
            # ---------- 补全 tra ----------
            if tra_len > 0 and tra_len < max_len:
                last_val = tra_list[-1]
                tra_list.extend(
                    [last_val] * (max_len - tra_len))
                mismatch_items.append(
                    f"{name}.tra({tra_len}->{max_len})")

        # 3. 提醒用户
        if mismatch_items:
            nm.post_notification(
                f"Band length mismatch detected. "
                f"Auto padded to {max_len}: "
                f"{', '.join(mismatch_items)}",
                status=nm.NotificationStatus.WARNING,
                duration=5)

    def prepare_simulation_environment(self):
        '''进入模拟前,进行模拟环境配置设定
        '''
        # 初始化进度条窗口
        self._progress_win = SimulationProgressWindow()
        # 如果存在上一次的'/Replicator'，就执行删除
        prim = self.stage.GetPrimAtPath('/Replicator')
        if prim:
            omni.kit.commands.execute('DeletePrims', paths=[Sdf.Path('/Replicator')], destructive=False)

        # 关闭可视化窗口
        self.viewport = vp_utils.get_active_viewport_window()
        self.viewport.visible = False

        # 色调映射改到clamp，跳过一切曝光调整
        self._settings.set('/rtx/post/tonemap/op', 0)
        # 取消gamma校正
        self._settings.set('/rtx/post/tonemap/enableSrgbToGamma', False)
        # 关闭fireflyFilter，“萤火虫”（Fireflies，即极亮且无法通过增加采样消除的孤立像素点），可能会对辐亮度进行截断
        self._settings.set('/rtx/pathtracing/fireflyFilter/enabled', False)
        # 关闭降噪
        # self._settings.set('/rtx/pathtracing/optixDenoiser/enabled', False)
        # 关闭matteObject
        self._settings.set('/rtx/matteObject/enabled', False)

    async def cleanup_simulation_environment(self, start_time):
        '''退出模拟后,恢复可视化环境设定
        '''
        # 计算并输出总耗时
        end_time = time.time()
        elapsed = end_time - start_time
        self._progress_win.set_elapsed_time(elapsed)

        # 开启可视化窗口
        self.viewport.visible = True
        # 切换回realtime渲染模式
        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action("omni.kit.viewport.actions", "set_renderer_rtx_realtime")
        action.execute()

        # 色调映射改到Aces，辐照度的gamma校正，曝光校正恢复默认的0.02
        self._settings.set('/rtx/post/tonemap/op', 6)
        self._settings.set('/rtx/post/tonemap/enableSrgbToGamma', True)
        self._settings.set('/rtx/post/tonemap/exposureTime', 0.02)
        # 关闭渲染结算模式
        self._settings.set("/rtx/pathtracing/settle/enabled", False)
        # 恢复渲染器的辐射度上限,默认为none
        self._settings.set("/rtx/pathtracing/maxRadiance", None)
        # 开启一般渲染优化设定
        self._settings.set('/rtx/pathtracing/fireflyFilter/enabled', True)
        # self._settings.set('/rtx/pathtracing/optixDenoiser/enabled', True)
        self._settings.set('/rtx/matteObject/enabled', True)

        # 显示原场景光源
        self.__set_original_lights_visibility(True)

        # 还原材质为RGB颜色
        for name, data in self.reflectance_data.items():
            original_color = data.get('display_color', [])
            shader_path = f"/World/Looks/{name}/Shader"
            shader_prim = rep.get.prims(path_pattern=str(shader_path))
            r_tuple = (original_color[0], original_color[1], original_color[2])
            with shader_prim:
                rep.modify.attribute(
                        name="inputs:diffuse_reflection_color",
                        value=rep.distribution.sequence([r_tuple]),
                        attribute_type="color3f"
                    )
                rep.modify.attribute(
                        name="inputs:subsurface_transmission_color",
                        value=rep.distribution.sequence([r_tuple]),
                        attribute_type="color3f"
                    )

        # 删除本次的stage中'/Replicator'
        rep.orchestrator.stop()
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()
        omni.kit.commands.execute('DeletePrims', paths=[Sdf.Path('/Replicator')], destructive=False)

        # 重开一次stage
        # for i in range(3):
        #     await omni.kit.app.get_app().next_update_async()  # 异步等待更新，防止UI崩溃
        # ctx = omni.usd.get_context()
        # ctx.reopen_stage()

    def create_and_setup_light(self):
        '''
        '''
        # 缓存场景需要隐藏的路径
        self.__cache_stage_elements()
        # 隐藏场景原光照
        self.__set_original_lights_visibility(False)

        # 模拟时光照舞台路径
        sun_light_path = '/Replicator/Sun_Light'
        sky_light_path = '/Replicator/Sky_Light'

        # 创建光源
        IlluminationUtils.create_light(sun_light_path, sky_light_path, self.params['direct_sun_intensity'], self.params['diffuse_sky_intensity'])
        # 设定光源属性
        zenith_and_azimuth = self.params['light_zenith_and_azimuth']
        IlluminationUtils.setup_sun_light_orient(sun_light_path, zenith_and_azimuth[0], zenith_and_azimuth[1])

    def __add_reflectance_panel(self, panel_size: float = 3.0, fill_factor: float = 0.6):
        '''引入反射板,并且计算传感器观测反射板的合适的位置

        Args:
            param self: 说明
            camera_path: 相机路径
            panel_size: 反射板的边长（米）
            fill_factor: 反射板占屏幕宽度的比例 (0.8 = 80%)
        Returns:
            焦距(mm);
        '''

        # 1. 获取相机当前的视场参数
        # horizontal_aperture 是 USD 相机的光圈宽度，focal_length 是焦距
        # 计算水平视场角 FOV (弧度)
        fov_rad = 2 * math.atan(self._horizontal_aperture_attr.Get() / (2 * self._focal_length_value_attr.Get()))

        # 2. 计算最佳定标距离 D
        # D = (板宽/2) / tan(视野/2 * 占比)
        distance = (panel_size / 2.0) / (fill_factor * math.tan(fov_rad / 2.0))

        # 3. 设置定标位姿
        # 相机放在 Z 轴上方 distance 处，低头看向 (0,0,0)
        calib_pos = (0, 0, distance)
        calib_target = (0, 0, 0)

        # 4. 设置定参考板位置
        reflectance_panel = rep.create.from_usd(REFLECTANCE_PANEL_ASSET)
        with reflectance_panel:
            rep.modify.pose(position=calib_target)
        return calib_pos, calib_target, reflectance_panel

    async def observe_reflectance_panel(self, reflectance_panel, calib_camera_pos, writer):

        # 1. 定标阶段，隐藏场景，添加反射板在/replicator下是显示的
        self.__set_scene_visibility(False)
        # 缓存原始相机属性
        original_translate_value = self.camera_translate_op.Get()
        original_aperture_value = self._horizontal_aperture_attr.Get()
        # 调整辐射定标传感器属性
        if self.params['sensor_type'] == 'Perspective':
            self.camera_translate_op.Set(calib_camera_pos)
        if self.params['sensor_type'] == 'Orthographic':
            # 定标专用水平光圈宽度
            # 由透视相机FOV逻辑（add_reflectance_panel）计算的水平光圈宽度不适用正射相机
            self._horizontal_aperture_attr.Set(100.0)
            # 定标专用高度，由透视相机FOV逻辑计算的高度不适用正射相机
            self.camera_translate_op.Set((0, 0, 50))
        custom_prefix = "radiometric_calibration"
        writer.set_custom_name(custom_prefix)
        # 步进定标模拟
        await rep.orchestrator.step_async()

        # 2. 定标结束，恢复场景，隐藏反射板，隐藏原场景光源
        self.__set_scene_visibility(True)
        with reflectance_panel:
            rep.modify.visibility(False)
        # 恢复传感器属性
        if self.params['sensor_type'] == 'Perspective':
            self.camera_translate_op.Set(original_translate_value)
        if self.params['sensor_type'] == 'Orthographic':
            self.camera_translate_op.Set(original_translate_value)
            self._horizontal_aperture_attr.Set(original_aperture_value)  # 恢复正常宽度

    def __cache_stage_elements(self):
        """在模拟开始前调用一次，缓存需要操作的节点"""
        self._cached_scene_prims = []
        self._cached_light_prims = []

        # 1. 缓存顶级场景节点
        root = self.stage.GetPseudoRoot()
        for prim in root.GetChildren():
            path = str(prim.GetPath())
            if not any(path.startswith(s) for s in self._system_paths):
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    self._cached_scene_prims.append(imageable)

        # 2. 缓存场景光源 (排除 Replicator 产生的光源)
        for prim in self.stage.Traverse():
            light_api = UsdLux.LightAPI(prim)
            if light_api:
                if not str(prim.GetPath()).startswith("/Replicator"):
                    self._cached_light_prims.append(UsdGeom.Imageable(prim))

    def __set_scene_visibility(self, visible: bool):
        """高效切换场景显隐"""
        for prims in self._cached_scene_prims:
            if visible:
                prims.MakeVisible()
            else:
                prims.MakeInvisible()

    def __set_original_lights_visibility(self, visible: bool):
        """高效切换原场景光源显隐"""
        for light in self._cached_light_prims:
            if visible:
                light.MakeVisible()
            else:
                light.MakeInvisible()

    def __calculate_observation_position(self):
        '''
        这里计算传感器的位置
        '''
        observation_position = []
        ground_footprint = []
        # 获取观测模式
        observation_type = self.params.get("optical_observation_type")
        if observation_type == 'Single Sampling':
            # 直接读取当前传感器的位置
            observation_position = [self.camera_translate_op.Get()]
            target_position = (0, 0, 0)

        elif observation_type == 'Constant Altitude Sampling':
            observation_position, ground_footprint, target_position = calculate_constant_altitude_sampling_waypoints(
                self.params['constant_altitude_sampling_start_point'], self.params['constant_altitude_sampling_end_point'], self.params['constant_altitude_sampling_flight_altitude'], self.params['optical_sensor_pixels'],
                self._focal_length_value_attr.Get(), self._horizontal_aperture_attr.Get(), self.params['constant_altitude_sampling_forward_and_side_overlap'], self.params.get("sensor_type")
            )

        elif observation_type == 'Semi-circular Sampling':
            observation_position, uav_position, target_position = calculate_semi_circular_sampling_waypoints(
                self.params['semicircular_sampling_distance'], self.params['semicircular_sampling_view_azimuth'], self.params['semicircular_sampling_zenith_range'],
                self.params['semicircular_sampling_zenith_step'], self.params['semicircular_sampling_observation_center']
            )
        elif observation_type == 'Omnidirectional Sampling':
            observation_position, ground_footprint, target_position = calculate_omnidirectional_sampling_waypoints(
                self.params['omnidirectional_sampling_distance'], self.params['omnidirectional_sampling_view_zenith'],
                self.params['omnidirectional_sampling_azimuth_step'], self.params['omnidirectional_sampling_observation_center']
            )
        return observation_position, target_position

    def __sensor_motion_control(self, observation_position, target_position, frame_idx):
        # 单次观测不改变位置
        observation_type = self.params['optical_observation_type']
        if observation_type == 'Single Sampling':
            pass
        else:
            camera_position = observation_position[frame_idx]  # 传感器位置
            target = target_position[frame_idx]  # 目标位置
            # self.camera_translate_op.Set(camera_position)
            CameraUtils.set_camera_pose_lookat_quat(self.stage, self.sensor_path, camera_position, target)

    def __radiometric_calibration(self, bands_data):
        '''
        '''
        # 辐射定标流程
        rc_pipeline = RadiometricCalibration(base_dir=self.intermediate_path)
        models = rc_pipeline.radiometric_calibration_pipeline(
            REFLECTANCE_VALUES, REFLECTANCE_KEYS, REFLECTANCE_SEMANTIC_TYPE, REFLECTANCE_SEMANTIC_DATA)  # 计算经验经验线性拟合参数
        # npytohdr
        npytohdr_pipeline = NpyToReflectanceHdrConverter(self.intermediate_path, self.result_path)
        npytohdr_pipeline.npy_to_multichannel_tiff(models, bands_data)

        # 针对半圆观测开启BRF曲线
        if self.params.get("optical_observation_type") == "Semi-circular Sampling":
            BRFCurveCalculation.main(self.result_path, self.params['semicircular_sampling_zenith_range'], self.params['semicircular_sampling_zenith_step'])
