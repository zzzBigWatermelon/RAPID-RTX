__all__ = ["SceneConstructionWindow"]

import omni.kit.commands
from pxr import Sdf, Gf, UsdGeom, UsdLux
import omni.ui as ui
from .window_style import get_style
from .window_combo_box_model import ComboBoxModel
import carb
from pathlib import Path
import asyncio
# 自定义功能
from rapid.Utility.window_components_combo_box_model import ComboBoxModel
from rapid.Utility.file_picker_to_UI_model import FilePickerHelper  # 文件选择器
from rapid.Utility.searchable_prim_picker import SearchablePrimPicker  # 舞台prim搜索选择功能
from rapid.Utility.illumination_utils import IlluminationUtils  # 照明设定
from .distribute_object import TerrainHeightSampler, ImportDataDistributor
from .scene_construction_utils import SceneConstructionUtils

LABEL_WIDTH = 120
SPACING = 4

# 定义路径
DATA_ROOT = Path(__file__).parent.parent.parent/'data'
VEGETATION_MODEL = str(DATA_ROOT/'Virtual Vegetation Model')
IMPORT_DISTRIBUTION_DEFAULT_FODER = str(DATA_ROOT/'example_position_folder')


class SceneConstructionWindow(ui.Window):
    """这个类设计窗口的具体层次结构"""

    def __init__(self, title: str, delegate=None, **kwargs):
        self.__label_width = LABEL_WIDTH

        super().__init__(title, **kwargs)

        # 放大镜图标的位置
        self.search_icon_path = str(DATA_ROOT / "search.svg")

        # 所有的 UI Model 数据持久化 (UI 清空时数据不会丢)
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
        self.frame.set_build_fn(self._build_fn)

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

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def _init_models(self):
        # 初始化专门存储模型的字典
        # UI模型直接存入字典，键名与最终输出数据的键名保持一致
        self.models = {}

        # -------------------------控制光照的UImodel-------------------------
        self.models["select_sun_light_path"] = ui.SimpleStringModel('/Light/Direct_Sun_Light')
        self.models["select_sky_light_path"] = ui.SimpleStringModel('/Light/Diffuse_Sky_Light')
        self.models["light_zenith_and_azimuth"] = [ui.SimpleFloatModel(v) for v in (45, 0)]
        self.models["sky_diffuse_fraction"] = ui.SimpleFloatModel(0.3)
        self.models["solar_energy_scale"] = ui.SimpleFloatModel(1)

        # -------------------------控制地形UImodel-------------------------
        self.models["terrain_type_model"] = ComboBoxModel("Simple Plane", "Randomly Generated Terrain", "Import Terrain Data (Raster)", "Import Terrain Data (Mesh)")
        self.models["terrain_extent_models"] = [ui.SimpleFloatModel(v) for v in (100, 100)]

        # -------------------------objects导入和选择的UImodel-------------------------
        # 初始化导入，把导入资产的 UI 模型传进去
        self.models["import_asset_model"] = ui.SimpleStringModel()  # 显示导入资产的路径
        self.import_asset_fn = FilePickerHelper(model_to_update=self.models["import_asset_model"],
                                                default_path=VEGETATION_MODEL,
                                                import_to_stage=True)
        # 搜索并选中的prim的舞台路径
        self.models["select_stage_prim_model"] = ui.SimpleStringModel()

        # ------------------------控制prim分布的UImodel-------------------------
        # 随机和均匀分布
        self.models["distribution_type_model"] = ComboBoxModel("Random", "Uniform", "Import Positon")
        self.models["instancing_mode_model"] = ComboBoxModel("PrimInstancing", "PointInstancer")
        self.models["distribution_number_model"] = ui.SimpleIntModel(1000)
        self.models["distribution_extent_model"] = [ui.SimpleFloatModel(v) for v in (100, 100)]

        # 导入分布位置
        self.models["import_positon_model_folder"] = ui.SimpleStringModel()  # 导入资产
        self.model_folder_path_model = FilePickerHelper(self.models["import_positon_model_folder"],
                                                        IMPORT_DISTRIBUTION_DEFAULT_FODER,
                                                        False)
        # 导入位置数据文件
        self.models["import_positon_data_file"] = ui.SimpleStringModel()
        self.positon_data_file_path_model = FilePickerHelper(model_to_update=self.models["import_positon_data_file"],
                                                             default_path=IMPORT_DISTRIBUTION_DEFAULT_FODER,
                                                             import_to_stage=False)
        # 导入DEM数据文件
        self.models["import_dem_mesh_file"] = ui.SimpleStringModel()
        self.dem_mesh_file_path_model = FilePickerHelper(model_to_update=self.models["import_dem_mesh_file"],
                                                         default_path=IMPORT_DISTRIBUTION_DEFAULT_FODER,
                                                         import_to_stage=True)

    def _on_stage_event(self, event: carb.events.IEvent):
        # 检查事件类型是否为 "Stage 已打开"
        if event.type == int(omni.usd.StageEventType.OPENED):
            carb.log_info("Detected Stage Opened event. Syncing UI parameters from JSON...")

            # 确保 models 已经初始化
            if hasattr(self, "models") and self.models:
                SceneConstructionUtils.read_json_data_to_UI(self.models)
                pass
            else:
                carb.log_warn("UI models not initialized yet, skipping sync.")

    def _build_fn(self):
        """
        组织窗口的主要组件,设定主要的ui框架,并将每个主要组件的细节构建逻辑移到另一个函数中。
        """
        # 如果窗口大小不合适，ScrollingFrame会添加滚动条
        with ui.ScrollingFrame():
            with ui.VStack(spacing=5, height=0):
                self._build_illumination_frame()
                self._build_terrain_frame()
                self._build_objects_frame()

                # The Go button
                # ui.Button("Scatter", clicked_fn=self._on_scatter)

    def _build_objects_frame(self):
        """引入模型并分散在stage中"""
        with ui.CollapsableFrame(title="Objects",
                                 name="groupFrame",
                                 height=0,
                                 style=get_style()):
            with ui.VStack(height=0, spacing=SPACING):
                with ui.CollapsableFrame(title="Import Objects",
                                         name="subFrame",
                                         height=0,
                                         collapsed=False,
                                         style=get_style(),
                                         horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                         vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    self._build_objects_import_model()

                with ui.CollapsableFrame(title="Distribution Objects",
                                         name="subFrame",
                                         height=0,
                                         collapsed=False,
                                         style=get_style(),
                                         horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                         vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    self._build_objects_distribution_model()

    def _build_objects_import_model(self):
        ''''''
        with ui.VStack(height=0, spacing=SPACING):
            with ui.HStack():
                ui.Label("Import Objects", name="Label", width=self.label_width, style=get_style())
                ui.StringField(model=self.models["import_asset_model"])
                # Button that puts the selection to the string field
                ui.Button(width=30, height=22, image_url=self.search_icon_path, clicked_fn=lambda: self.import_asset_fn.select_file_or_folder())
            with ui.HStack():
                ui.Label("Select Prim", width=self.label_width, style=get_style())
                ui.StringField(model=self.models["select_stage_prim_model"])
                # 在舞台上寻找prim
                ui.Button(width=30, height=22, image_url=self.search_icon_path,
                          clicked_fn=lambda: self._on_open_picker_clicked(self.models["select_stage_prim_model"], [UsdGeom.Xform, UsdGeom.Mesh]))

    def _build_objects_distribution_model(self):
        ''''''
        with ui.VStack(height=0, spacing=SPACING):
            with ui.HStack():
                ui.Label("Distribution Type", name="Distribution_name", width=self.label_width)
                ui.ComboBox(self.models["distribution_type_model"])

            # 随机分布和均匀分布方式
            self.random_uniform_distribution = ui.VStack(height=0, spacing=SPACING, visible=True)
            with self.random_uniform_distribution:
                with ui.HStack():
                    ui.Label("Instancing Type", width=self.label_width)
                    ui.ComboBox(self.models["instancing_mode_model"])
                with ui.HStack():
                    ui.Label("Number",  width=self.label_width)
                    ui.IntField(model=self.models["distribution_number_model"])
                    ui.Label("Extent X(m)", width=self.label_width)
                    ui.FloatField(model=self.models["distribution_extent_model"][0])
                    ui.Label("Extent Y(m)", width=self.label_width)
                    ui.FloatField(model=self.models["distribution_extent_model"][1])
                # The Go button
                ui.Button("Distribute Object", clicked_fn=self._on_distribute_object)

            # 导入位置文件的分布方式
            self.import_distribution = ui.VStack(height=0, spacing=SPACING, visible=False)
            with self.import_distribution:
                with ui.HStack():
                    ui.Label("Model Folder", name="Label", width=self.label_width, style=get_style())
                    ui.StringField(model=self.models["import_positon_model_folder"])
                    # Button that puts the selection to the string field
                    ui.Button(width=30, height=22, image_url=self.search_icon_path, clicked_fn=lambda: self.model_folder_path_model.select_file_or_folder())
                with ui.HStack():
                    ui.Label("Data File", name="Label", width=self.label_width, style=get_style())
                    ui.StringField(model=self.models["import_positon_data_file"])
                    # Button that puts the selection to the string field
                    ui.Button(width=30, height=22, image_url=self.search_icon_path, clicked_fn=lambda: self.positon_data_file_path_model.select_file_or_folder())
                with ui.HStack():
                    ui.Label("DEM File", name="Label", width=self.label_width, style=get_style())
                    ui.StringField(model=self.models["import_dem_mesh_file"])
                    # Button that puts the selection to the string field
                    ui.Button(width=30, height=22, image_url=self.search_icon_path, clicked_fn=lambda: self.dem_mesh_file_path_model.select_file_or_folder())

                ui.Button("Import Distribute", clicked_fn=self.on_import_distribution)

        # 绑定下拉选择框的回调函数,控制不同参数的显示
        if not hasattr(self, "_sampling_type_handler"):
            self._sampling_type_handler = self.models["distribution_type_model"].add_item_changed_fn(self._on_distribution_type_changed)
        # 这里的逻辑是为了防止重新构建 UI 时状态丢失，手动触发一次同步
        self._on_distribution_type_changed(self.models["distribution_type_model"], None)

    def _build_terrain_frame(self):
        ''''''
        with ui.CollapsableFrame(title="Terrain",
                                 name="groupFrame",
                                 height=0,
                                 style=get_style()):
            with ui.VStack(height=0, spacing=SPACING):
                with ui.CollapsableFrame(title="PLANE",
                                         name="subFrame",
                                         height=0,
                                         collapsed=False,
                                         style=get_style(),
                                         horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                         vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    self._build_terrain_subFrame()

    def _build_terrain_subFrame(self):
        with ui.VStack(height=0, spacing=SPACING):
            with ui.HStack():
                ui.Label("Terrain Type", width=self.label_width)
                ui.ComboBox(self.models["terrain_type_model"])
            with ui.HStack():
                ui.Label("Extent X(m)", width=self.label_width)
                ui.FloatField(model=self.models["terrain_extent_models"][0])
                ui.Label("Extent Y(m)", width=self.label_width)
                ui.FloatField(model=self.models["terrain_extent_models"][1])

            # The Go button
            ui.Button("Create Terrain", clicked_fn=self._on_terrain_cteate)

    def _build_illumination_frame(self):
        ''''''
        with ui.CollapsableFrame(title="Illumination & Atmosphere",
                                 name="groupFrame",
                                 height=0,
                                 style=get_style()):
            with ui.VStack(height=0, spacing=SPACING):
                with ui.CollapsableFrame(title="Illumination",
                                         name="subFrame",
                                         height=0,
                                         collapsed=False,
                                         style=get_style(),
                                         horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                         vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    self._build_illumination_subFrame()

    def _build_illumination_subFrame(self):
        with ui.VStack(height=0, spacing=SPACING):
            with ui.HStack():
                ui.Label("Select Sun Light", width=self.label_width, style=get_style())
                ui.StringField(model=self.models["select_sun_light_path"])
                # 在舞台上寻找sun_light
                ui.Button(width=30, height=22, image_url=self.search_icon_path,
                          clicked_fn=lambda: self._on_open_picker_clicked(self.models["select_sun_light_path"], [UsdLux.DiskLight]))
            with ui.HStack():
                ui.Label("Select Sky Light", width=self.label_width, style=get_style())
                ui.StringField(model=self.models["select_sky_light_path"])
                # 在舞台上寻找sky_light
                ui.Button(width=30, height=22, image_url=self.search_icon_path,
                          clicked_fn=lambda: self._on_open_picker_clicked(self.models["select_sky_light_path"], [UsdLux.DomeLight]))
            with ui.HStack():
                ui.Label("Sun Zenith [°]",  width=self.label_width)
                ui.FloatField(model=self.models["light_zenith_and_azimuth"][0])
                ui.Label("Sun Azimuth [°]", width=self.label_width)
                ui.FloatField(model=self.models["light_zenith_and_azimuth"][1])
            with ui.HStack():
                ui.Label("Sky Diffuse Fraction",  width=self.label_width)
                ui.FloatField(model=self.models["sky_diffuse_fraction"])
                ui.Label("Solar Energy Scale", width=self.label_width)
                ui.FloatField(model=self.models["solar_energy_scale"])
            ui.Button("Create Light", clicked_fn=self.on_create_light)

    def _on_terrain_cteate(self):
        # 读取当前窗口中的数据
        data = {
            "terrain_type": self.models["terrain_type_model"].get_current_item().as_string,
            "terrain_extent": [m.as_float for m in self.models["terrain_extent_models"]],
        }

        # 创建地形
        terrain_type = data["terrain_type"]
        terrain_extent = data["terrain_extent"]
        if terrain_type == 'Simple Plane':
            self._cteate_simple_terrain(terrain_extent)

    def _cteate_simple_terrain(self, terrain_extent):
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

        # 保存当前窗口数据到simulation_parameters.json文件
        SceneConstructionUtils.save_UI_data_to_json(self.models)

    def _on_distribution_type_changed(self, model, item):
        """当下拉菜单选择改变时触发"""
        # 必须传递 None 和 0, 否则会报 missing arguments 错误
        index = model.get_item_value_model(None, 0).as_int

        # 根据index改变UI的显隐
        if index == 0 or index == 1:
            self.random_uniform_distribution.visible = True
            self.import_distribution.visible = False
        elif index == 2:
            self.random_uniform_distribution.visible = False
            self.import_distribution.visible = True

    def _on_open_picker_clicked(self, target_model, type_filter):
        """
        通用的弹窗触发函数
        target_model: 点击确定后要更新的那个 model
        type_filter: 该行对应的搜索过滤类型
        """
        SearchablePrimPicker(
            title="Pick a Prim",
            type_filter=type_filter,
            # 使用 lambda 捕获 target_model，这样回调时就知道改哪一个
            on_select_fn=lambda path: self._on_prim_selected_callback(path, target_model)
        )

    def _on_prim_selected_callback(self, path, target_model):
        """通用选择回调"""
        target_model.as_string = path
        print(f"已更新路径为: {path}")

    def _on_distribute_object(self):
        # 读取当前distribute_object窗口中的数据
        select_prim_path = self.models["select_stage_prim_model"].as_string
        dist_type = self.models["distribution_type_model"].get_current_item().as_string
        dist_num = self.models["distribution_number_model"].as_int
        dist_extent = [m.as_float for m in self.models["distribution_extent_model"]]
        instancing_mode = self.models["instancing_mode_model"].get_current_item().as_string

        # 实例化 TerrainHeightSampler (传入参数)
        sampler_task = TerrainHeightSampler(
            prim_path=[select_prim_path],
            dist_type=dist_type,
            num=dist_num,
            extent=dist_extent,
            mode=instancing_mode  # 传入实例化的模式
        )

        asyncio.ensure_future(sampler_task.run())

        # 保存当前窗口数据到simulation_parameters.json文件
        SceneConstructionUtils.save_UI_data_to_json(self.models)

    def on_import_distribution(self):
        model_folder_path = self.models["import_positon_model_folder"].as_string
        data_file_path = self.models["import_positon_data_file"].as_string
        instancing_mode = self.models["instancing_mode_model"].get_current_item().as_string
        sampler_task = ImportDataDistributor(model_folder_path, data_file_path, instancing_mode)
        asyncio.ensure_future(sampler_task.run())

        # 保存当前窗口数据到simulation_parameters.json文件
        SceneConstructionUtils.save_UI_data_to_json(self.models)

    def on_create_light(self):
        # 获取UI数据
        sun_light_path = self.models["select_sun_light_path"].as_string
        sky_light_path = self.models["select_sky_light_path"].as_string
        light_zenith_and_azimuth = [m.as_float for m in self.models["light_zenith_and_azimuth"]]

        # 创建光源
        sky_diffuse_fraction = self.models["sky_diffuse_fraction"].as_float
        solar_energy_scale = self.models["solar_energy_scale"].as_float
        IlluminationUtils.create_light(sun_light_path, sky_light_path, light_zenith_and_azimuth[0], sky_diffuse_fraction, solar_energy_scale)
        # 设定光源方向
        IlluminationUtils.setup_sun_light_orient(sun_light_path, light_zenith_and_azimuth[0], light_zenith_and_azimuth[1])
        # 可视化光源
        IlluminationUtils.draw_illumination_visualization(light_zenith_and_azimuth[0], light_zenith_and_azimuth[1])
        # 保存当前窗口数据到simulation_parameters.json文件
        SceneConstructionUtils.save_UI_data_to_json(self.models)

    def get_UI_data_to_simulation(self):
        return {
            "light_zenith_and_azimuth": [i.as_float for i in self.models["light_zenith_and_azimuth"]],
            "direct_sun_intensity": self.models["sky_diffuse_fraction"].as_float,
            "diffuse_sky_intensity": self.models["solar_energy_scale"].as_float
        }
