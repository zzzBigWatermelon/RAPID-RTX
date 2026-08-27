__all__ = ["ReflectanceDatabaseWindow"]

from pxr import Sdf
import omni.ui as ui
from isaacsim.gui.components.style import get_style
import asyncio
import omni.kit.app
import omni.usd
import carb
from pathlib import Path
import numpy as np
import omni.kit.notification_manager as nm
# 自定义模块
from rapid.Utility.window_components_combo_box_model import ComboBoxModel
from rapid.Utility import project_validity_check
from .reflectanceDatabase_utils import parse_str_to_float_list, parse_UI_data_to_dic, get_spectrum_data_for_bands, read_spectra_file_path, save_UI_data_to_json, parse_json_to_UI_format
from .stage_material_manager import updata_stage_materials, delete_stage_material
from .spectral_preview_window import SpectralPreviewWindow
from .spectral_database_table_model import TableModel, TableDelegate

LABEL_HEIGHT = 24
LABEL_WIDTH = 120
SPACING = 8

# 颜色定义
COLOR_NORMAL = 0xFF222222      # 普通背景 (深灰)
COLOR_SELECTED = 0xFFD06020    # 选中高亮 (亮蓝色)
COLOR_BORDER = 0xFF444444


class ReflectanceDatabaseWindow(ui.Window):
    """这个类设计窗口的具体层次结构"""

    def __init__(self, title: str, delegate=None, **kwargs):

        super().__init__(title, **kwargs)

        # 所有的 UI Model 数据持久化 (UI 清空时数据不会丢)
        self._init_models()

        # 监听新打开Stage事件
        self._usd_context = omni.usd.get_context()
        # 订阅事件流 (OPENED, SAVED, CLOSED 等)
        self._stage_event_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._refresh_UI_data_on_stage_open
        )

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_UI_structure)

    def _init_models(self):
        # 初始化专门存储模型的字典
        # UI模型直接存入字典，键名与最终输出数据的键名保持一致
        self.models = {}

        # 输入波段信息的UI
        self.models["bands"] = ui.SimpleStringModel('400:1,500:1,600:1,700:1')
        self.models["bands_start"] = ui.SimpleFloatModel(600.0)
        self.models["bands_end"] = ui.SimpleFloatModel(900.0)
        self.models["bands_number"] = ui.SimpleIntModel(10)
        self.models["bands_width"] = ui.SimpleFloatModel(1.0)
        self.models["bands_overwrite_model"] = ui.SimpleBoolModel(True)
        self.models["bands_append_model"] = ui.SimpleBoolModel(False)
        self._define_bands_window = None  # 用于存储新define_bands窗口实例

        # 选择输入光谱数据的UI
        self.models["import_data_model"] = ComboBoxModel("From Database", "From Models", "Manual Input")

        # 从数据文件中导入数据的UI
        self.models["from_database_window"] = None  # 用于存储新database_windo窗口实例

        # 手动输入的name、反射率、透射率的默认值
        self.models['manual_input_default_value_name'] = ui.SimpleStringModel('Name')
        self.models['manual_input_default_value_reflectance'] = ui.SimpleStringModel('0.1, 0.1, 0.1, 0.1105, 0.1105, 0.1105, 0.1211')
        self.models['manual_input_default_value_transmittance'] = ui.SimpleStringModel('0.1, 0.1, 0.1, 0.1105, 0.1105, 0.1105, 0.1211')

        # 定义默认对应的光谱名称,与默认数据库下的文件名一致
        self.models['default_spectra_data_file'] = ['birch_leaf_green.csv', 'birch_branch.csv', 'soil_dry.csv']
        # 反射率数据表Model
        self.models['data_table_model'] = TableModel(
            ["Name", "Reflectance ", "Transmittance", "Display Color"],
            [
                ["birch_leaf_green", "0.0410, 0.0425, 0.0585, 0.1075", "0.0005, 0.0106, 0.0563, 0.1220", "0.5,0.5,0.5"],
                ["birch_branch", "0.0692, 0.0798, 0.1052, 0.1768", "0, 0, 0, 0", "0.5,0.5,0.5"],
                ["soil_dry", "0.0483, 0.0572, 0.0804, 0.1326", "0, 0, 0, 0 ", "0.5,0.5,0.5"]
            ])
        # 创建 Delegate，把 Model 传进去
        self._delegate = TableDelegate(self.models['data_table_model'])

    @property
    def ref_data_model(self):
        '''方便外部调用UI数据'''
        return self.models['data_table_model']

    @property
    def bands_data_model(self):
        '''方便外部调用UI数据'''
        return self.models["bands"]

    def _refresh_UI_data_on_stage_open(self, event: carb.events.IEvent):
        """回调：当舞台打开时，从 JSON 加载反射率数据库内容"""
        if event.type != int(omni.usd.StageEventType.OPENED):
            return

        # 确定要读取的文件路径
        project_params_path = Path(project_validity_check.get_folder("parameters")) / 'simulation_parameters.json'
        default_params_path = Path(__file__).parent.parent.parent / 'data' / 'simulation_parameters.json'
        target_file = project_params_path if project_params_path.exists() else default_params_path

        # 执行加载
        carb.log_info(f"Loading reflectance data from: {target_file}")
        bands_data, new_data = parse_json_to_UI_format(target_file)

        # 更新 UI Model
        self.models['bands'].set_value(bands_data)
        self.models['data_table_model'].reset_data(new_data)

    def _build_UI_structure(self):
        """
        组织窗口的主要组件,设定主要的ui框架,并将每个主要组件的细节构建逻辑移到另一个函数中。
        """
        # 如果窗口大小不合适,ScrollingFrame会添加滚动条
        with ui.ScrollingFrame():
            with ui.VStack(spacing=5):
                with ui.Frame(height=30):
                    self.build_manage_bands()
                with ui.Frame(height=30):
                    self.build_import_data_structure()
                with ui.Frame(height=300):
                    self._build_database_table()

                # 添加功能按钮
                with ui.HStack(height=30):
                    ui.Spacer()
                    ui.Button("Refresh",  width=180, clicked_fn=self._on_refresh_clicked)
                    ui.Button("Delete", clicked_fn=self._on_del_clicked)
                    ui.Button("Plot",  width=180, clicked_fn=None)
                    ui.Spacer()

    def build_manage_bands(self):
        '''对波段的信息操作UI'''
        with ui.HStack(height=LABEL_HEIGHT):
            ui.Label("Spectral Bands:", name="Spectral Bands", height=LABEL_HEIGHT, style=get_style())
            ui.StringField(model=self.models["bands"])

            # 定义波段列表，弹出一个新窗口
            ui.Spacer(width=10)
            ui.Button("Define Bands", clicked_fn=self.on_define_bands_click)

    def on_define_bands_click(self):
        """弹出新define_bands窗口"""
        # 创建新窗口
        self._define_bands_window = ui.Window("Define New Bands", width=600, height=650)

        # 互斥逻辑：勾选一个，取消另一个
        self.models["bands_overwrite_model"].add_value_changed_fn(self.on_bands_overwrite_check)
        self.models["bands_append_model"].add_value_changed_fn(self.on_bands_append_check)

        # 窗口结构
        self.build_define_bands_window()

    def on_bands_overwrite_check(self, m):
        if m.as_bool:
            self.models["bands_append_model"].set_value(False)

    def on_bands_append_check(self, m):
        if m.as_bool:
            self.models["bands_overwrite_model"].set_value(False)

    def build_define_bands_window(self):
        '''新的define_bands_window窗口
        '''
        with self._define_bands_window.frame:
            with ui.VStack(spacing=SPACING):
                with ui.HStack():
                    ui.Label("Band Start [nm]:", width=LABEL_WIDTH)
                    ui.FloatField(model=self.models["bands_start"])
                with ui.HStack():
                    ui.Label("Band End [nm]:", width=LABEL_WIDTH)
                    ui.FloatField(model=self.models["bands_end"])
                with ui.HStack():
                    ui.Label("Band Number", width=LABEL_WIDTH)
                    ui.IntField(model=self.models["bands_number"])
                with ui.HStack():
                    ui.Label("Band Width", width=LABEL_WIDTH)
                    ui.FloatField(model=self.models["bands_width"])

                ui.Spacer(height=10)
                ui.Line(style={"color": 0x33FFFFFF})  # 分割线

                # --- 勾选框区 ---
                with ui.HStack():
                    ui.CheckBox(model=self.models["bands_overwrite_model"])
                    ui.Label("Overwrite existing bands")
                    ui.Spacer(width=10)
                    ui.CheckBox(model=self.models["bands_append_model"])
                    ui.Label("Append to existing bands")
                ui.Spacer(height=20)

                # --- 按钮区 ---
                with ui.HStack(spacing=15, height=30):
                    ui.Button("Confirm", clicked_fn=self.on_bands_confirm_click, style={"background_color": 0xFF448844})
                    ui.Button("Cancel", clicked_fn=self.on_bands_cancel_click)

    def on_bands_confirm_click(self):
        '''
        '''
        # 获取bands计算参数
        bands_start = self.models["bands_start"].as_float
        bands_end = self.models["bands_end"].as_float
        bands_number = self.models["bands_number"].as_int
        bands_width = self.models["bands_width"].as_float

        # 计算新的参数
        bands_str = ''
        if bands_number > 1:
            # 生成等间距数字序列
            bands_array = np.linspace(bands_start, bands_end, bands_number)
            bands_str = ", ".join([f"{x:.2f}:{bands_width}" for x in bands_array])
        elif bands_number == 1:
            bands_str = f"{bands_start:.2f}"

        # 将数值返回UI中
        if self.models["bands_overwrite_model"].as_bool:
            # 覆盖
            self.models["bands"].set_value(bands_str)
        else:
            # 追加
            current = self.models["bands"].as_string
            if current:
                self.models["bands"].set_value(f"{current}, {bands_str}")
            else:
                self.models["bands"].set_value(bands_str)

        self._define_bands_window.visible = False

    def on_bands_cancel_click(self):
        self._define_bands_window.visible = False

    def build_import_data_structure(self):
        '''对反射率数据库的增删改操作'''
        with ui.VStack(height=0, spacing=SPACING):
            with ui.HStack():
                ui.Label("Import Data Model", style=get_style())
                ui.ComboBox(self.models["import_data_model"])

            # 不同数据输入模式
            # 从数据库导入
            self._from_database = ui.VStack(visible=True)
            with self._from_database:
                with ui.HStack():
                    ui.Spacer()
                    ui.Button("From Database",  width=180, clicked_fn=self._on_from_database_clicked)
                    ui.Spacer()

            # 从模型库导入
            self._from_models = ui.VStack(visible=False)
            with self._from_models:
                with ui.HStack():
                    ui.Spacer()
                    ui.Button("Lambertian Models...",  width=180, clicked_fn=self._on_lambertian_models_clicked)
                    ui.Button("Non-Lambertian Models...",  width=180, clicked_fn=self._on_non_lambertian_models_clicked)
                    ui.Spacer()

            # 手动输入数据
            self._manual_input = ui.VStack(visible=False)
            with self._manual_input:
                self.build_import_data_from_manual()

        # 绑定下拉选择框的回调函数,控制不同参数的显示
        if not hasattr(self, "_sampling_type_handler"):
            self._sampling_type_handler = self.models["import_data_model"].add_item_changed_fn(self._on_sampling_type_changed)
        # 这里的逻辑是为了防止重新构建 UI 时状态丢失，手动触发一次同步
        self._on_sampling_type_changed(self.models["import_data_model"], None)

    def _on_sampling_type_changed(self, model, item):
        """当下拉菜单选择改变时触发"""
        # 必须传递 None 和 0，否则会报 missing arguments 错误
        index = model.get_item_value_model(None, 0).as_int

        # 根据index改变UI的显隐
        if index == 0:  # Orthographic
            self._from_database.visible = True
            self._from_models.visible = False
            self._manual_input.visible = False
        elif index == 1:
            self._from_database.visible = False
            self._from_models.visible = True
            self._manual_input.visible = False
        elif index == 2:
            self._from_database.visible = False
            self._from_models.visible = False
            self._manual_input.visible = True

    def _on_from_database_clicked(self):
        '''
        '''
        if self.models["from_database_window"]:
            self.models["from_database_window"].destroy()
        # 光谱预览窗口
        self.models["from_database_window"] = SpectralPreviewWindow(self.models["bands"], self.models['data_table_model'])

    def _on_lambertian_models_clicked(self):
        pass

    def _on_non_lambertian_models_clicked(self):
        pass

    def build_import_data_from_manual(self):
        '''对反射率数据库的增删改操作'''
        with ui.HStack():
            ui.Label("Name:", name="database_object_name", width=LABEL_WIDTH, style=get_style())
            ui.StringField(model=self.models['manual_input_default_value_name'])
            ui.Label("Ref:", name="database_object_ref", width=LABEL_WIDTH, style=get_style())
            ui.StringField(model=self.models['manual_input_default_value_reflectance'])
            ui.Label("Tra:", name="database_object_tra", width=LABEL_WIDTH, style=get_style())
            ui.StringField(model=self.models['manual_input_default_value_transmittance'])

            # The Go button
            ui.Spacer(width=10)
            ui.Button("Add", clicked_fn=self._on_manual_input_add_clicked)

    def _on_manual_input_add_clicked(self):
        '''新增反射率数据行并同步全量数据到 Stage'''
        # 校验当前输入框内名称的有效性
        new_name = self.models['manual_input_default_value_name'].as_string.strip()
        validity = self.name_validity_verification(new_name)
        if validity is False:
            return

        # 获取输入框当前的数据字符串
        ref_str = self.models['manual_input_default_value_reflectance'].as_string
        tra_str = self.models['manual_input_default_value_transmittance'].as_string

        # 保证反射率和透射率波段数量一致,校验波段数量是否相同
        ref_str, tra_str = self._fix_ref_tra_band_mismatch(ref_str, tra_str)

        # 将输入框中的内容更新到反射率的表格中
        self.models['data_table_model'].add_row(new_name, ref_str, tra_str)

    def name_validity_verification(self, new_name):
        '''新增行的材质名进行校验,检验是否存在相同材质名'''

        # 1. 校验：材质名不能为空
        if not new_name:
            nm.post_notification("name cannot be empty", status=nm.NotificationStatus.WARNING, duration=5)
            return False

        # 2. 校验：材质名不能以数字开头
        if new_name[0].isdigit():
            nm.post_notification(
                f"Addition failed: Name '{new_name}' cannot start with a number.",
                status=nm.NotificationStatus.WARNING,
                duration=5
            )
            return False

        # 3. 校验：检查是否已存在相同名字
        existing_items = self.models['data_table_model'].get_item_children(None)
        for item in existing_items:
            if item.name_model.as_string == new_name:
                # 弹出你要求的提示
                nm.post_notification(
                    "Objects with the same name already exist.",  
                    status=nm.NotificationStatus.WARNING,  # 蓝色提醒
                    duration=5
                )
                return False

    def _fix_ref_tra_band_mismatch(self, ref_str, tra_str):
        """
        同步反射率(Ref)和透射率(Tra)的长度，使其与主配置 bands 的数量严格一致。
        逻辑：长度不足补 0, 长度超出则裁剪。
        """
        # 1. 获取目标波段基准数量 (n_target)
        bands_str = self.models["bands"].as_string
        bands_list = parse_str_to_float_list(bands_str)
        n_target = len(bands_list)

        if n_target == 0:
            # 如果主波段配置为空，无法对齐，直接返回原数据并告警
            nm.post_notification(
                "Band definition is empty. Cannot sync data.",
                status=nm.NotificationStatus.ERROR)
            return ref_str, tra_str

        # 2. 解析输入数据
        ref_list = parse_str_to_float_list(ref_str)
        tra_list = parse_str_to_float_list(tra_str)

        # 3. 定义内部对齐工具函数
        def sync_length(data_list, target_count):
            current_count = len(data_list)
            if current_count < target_count:
                # 补全逻辑
                return data_list + [0.0] * (target_count - current_count), True
            elif current_count > target_count:
                # 裁剪逻辑
                return data_list[:target_count], True
            return data_list, False

        # 4. 执行对齐
        new_ref_list, ref_changed = sync_length(ref_list, n_target)
        new_tra_list, tra_changed = sync_length(tra_list, n_target)

        # 5. 提示用户 (如果发生了任何改变)
        if ref_changed or tra_changed:
            msg = f"Data synced to {n_target} bands. "
            if ref_changed:
                msg += f"Ref:{len(ref_list)}->{n_target}. "
            if tra_changed:
                msg += f"Tra:{len(tra_list)}->{n_target}."

            nm.post_notification(
                msg,
                status=nm.NotificationStatus.WARNING,
                duration=5
            )

        # 6. 转回字符串 (格式化保留 4 位小数以保持紧凑且精确)
        ref_str_new = ", ".join([f"{x:.4f}".rstrip('0').rstrip('.') for x in new_ref_list])
        tra_str_new = ", ".join([f"{x:.4f}".rstrip('0').rstrip('.') for x in new_tra_list])

        return ref_str_new, tra_str_new

    def _build_database_table(self):
        ui.Separator(height=5)
        with ui.ScrollingFrame(
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON
        ):

            # 构建表格样式,地物名称,反射率,透射率
            self._tree_view = ui.TreeView(
                self.models['data_table_model'],
                delegate=self._delegate,
                root_visible=False,
                header_visible=True,
                columns_resizable=True,
                # 稍微调整一下列宽比例,
                selection=[],
                column_widths=[ui.Fraction(1.5), ui.Fraction(1.5), ui.Fraction(1)]
            )
        self._delegate.bind_tree_view(self._tree_view)

    def _on_refresh_clicked(self):
        '''根据现有表格中的name和.models["bands"]中的波长,再重新更新一边表格的反射率/透射率'''
        self.refresh_button_fn()
        # 解析反射率的表格内所有数据（包括刚才新增的）
        all_window_data = parse_UI_data_to_dic(self.models['data_table_model'], self.models["bands"])
        # 调用修改后的接口，将所有数据循环添加/更新到stage
        asyncio.ensure_future(updata_stage_materials(all_window_data['ref_tra_data']))
        # 提示stage中成功对应的材质
        nm.post_notification(
            "Reflectance data refresh complete",
            status=nm.NotificationStatus.INFO,
            duration=5
        )

        # 更新simulation_parameters.json文件
        save_UI_data_to_json(self.models["bands"], self.models['data_table_model'])

    def refresh_button_fn(self):
        '''根据现有表格中的name和.models["bands"]中的波长,再重新更新一边表格
        只从数据库中更新反射率和透射率列,displaycolor使用原表格中的数据
        '''
        # 获取波长:波段宽度数据、反射率表格数据、反射率数据的name
        band_str = self.models["bands"].as_string
        model = self.models['data_table_model']
        display_colors = [item.display_color_model.as_string for item in model._items]
        all_names = [item.name_model.as_string for item in model._items]

        # 获取simulation_parameters.json文件路径,返回其中的的光谱信息内容
        default_parameters_path = Path(__file__).parent.parent.parent/'data'
        project_parameters_path = Path(project_validity_check.get_folder("parameters"))
        spectra_database = read_spectra_file_path(default_parameters_path/"simulation_parameters.json",
                                                  project_parameters_path/"simulation_parameters.json")
        # 存贮新数据用于重置表格
        new_data = []
        for index, spectra_name in enumerate(all_names):
            # 从索引中拿到文件名 (例如: 'birch_leaf.csv')
            csv_filename = spectra_database[spectra_name]["csv_path"]
            # 先尝试拼接项目路径，如果项目路径下不存在，则转向默认路径
            target_csv_path = project_parameters_path / csv_filename
            if not target_csv_path.exists():
                target_csv_path = default_parameters_path / csv_filename
            # 调用之前写在 utils 里的计算函数，计算反射率
            ref_str, tra_str = get_spectrum_data_for_bands(str(target_csv_path), band_str)
            new_data.append([spectra_name, ref_str, tra_str, display_colors[index]])

        # 更新 UI 表格
        self.models['data_table_model'].reset_data(new_data)

    def _on_del_clicked(self):
        items_to_delete = [item for item in self.models['data_table_model'].get_item_children(None) if item.is_selected_model.as_bool]
        for item in items_to_delete:
            mat_name = item.name_model.as_string
            # 调用接口从场景删除
            delete_stage_material(mat_name)

        self.models['data_table_model'].remove_items(items_to_delete)

    def _on_plot_clicked(self):
        """读取选中行数据并弹出新窗口画图"""
        model = self.models['data_table_model']

        # 1. 获取选中的行
        selected_item = None
        for item in model._items:
            if item.is_selected_model.as_bool:
                selected_item = item
                break

        if not selected_item:
            nm.post_notification("Please select a row in the table.", status=nm.NotificationStatus.WARNING)
            return

        # 2. 解析数据
        name = selected_item.name_model.as_string

        def parse_str(s):
            return [float(x.strip()) for x in s.split(',') if x.strip()]

        ref_list = parse_str(selected_item.ref_value_model.as_string)
        tra_list = parse_str(selected_item.tra_value_model.as_string)

        # 3. 创建并显示新窗口
        # 我们将引用存在 self 中，防止窗口对象被垃圾回收
        # self._plot_window = SpectralPlotWindow(name, ref_list, tra_list)

        # 如果希望每次点击都产生一个独立的新窗口，而不覆盖旧的：
        # if not hasattr(self, "_plot_windows"): self._plot_windows = []
        # self._plot_windows.append(SpectralPlotWindow(name, ref_list, tra_list))

    def destroy(self):
        # It will destroy all the children
        self.models['data_table_model'] = None
        self._delegate = None
        self._stage_event_sub = None
        super().destroy()
