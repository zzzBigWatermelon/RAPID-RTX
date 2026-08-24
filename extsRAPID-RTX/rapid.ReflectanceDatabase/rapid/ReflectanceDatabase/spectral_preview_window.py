import omni.ui as ui
import os
import shutil
import json
from pathlib import Path
import omni.kit.notification_manager as nm
from omni.kit.window.filepicker import FilePickerDialog
from rapid.Utility import project_validity_check  # 项目有效性检查
from .reflectanceDatabase_utils import save_UI_data_to_json, read_csv_file, get_spectrum_data_for_bands, read_spectra_file_path


class SpectralPreviewWindow(ui.Window):
    def __init__(self, bands_model, data_table_model):
        super().__init__("Spectral Database Preview", width=800, height=600)
        # 检查项目文件环境完整性
        if not project_validity_check.get_current_usd_path() or not project_validity_check.quick_project_check():
            return
        # 初始化参数
        self.bands_model = bands_model
        self.data_table_model = data_table_model

        # 路径初始化
        DATA_ROOT = Path(__file__).parent.parent.parent/'data'  # 默认数据库
        PROJECT_PARAMETERS_PATH = Path(project_validity_check.get_folder("parameters"))
        self.default_parameters_path = DATA_ROOT / "simulation_parameters.json"
        self.project_parameters_path = PROJECT_PARAMETERS_PATH / "simulation_parameters.json"
        self.project_spectra_database_path = PROJECT_PARAMETERS_PATH / "spectra_database"  # 项目文件数据库
        self.project_spectra_database_path.mkdir(parents=True, exist_ok=True)  # 不存在就创建
        # 初始化核心变量
        self.database_index = {}
        self.current_spectrum_name = "No Data Loaded"
        # 窗口提前声明
        self._list_stack = None
        self._import_spectral_file_picker = None

        # 执行扫描光谱文件逻辑
        self.database_index.clear()
        self.database_index = read_spectra_file_path(self.default_parameters_path,
                                                     self.project_parameters_path)
        # 构建UI
        self._build_spectral_preview_window()

        # 启动时自动显示第一条光谱曲线
        self._load_default_item()

    def _build_spectral_preview_window(self):
        with self.frame:
            with ui.VStack(spacing=10):
                # --- 顶部工具栏 ---
                self._build_top_toolbar()

                # 模拟数据库列表和可视化区域
                with ui.HStack():
                    # --- 左侧：模拟数据库列表 ---
                    self._build_spectral_databases_list()
                    # --- 右侧：可视化区域 ---
                    self._build_spectral_visualization()

    def _build_top_toolbar(self):
        with ui.HStack(height=40, spacing=10):
            ui.Button("Import", clicked_fn=self._on_import_clicked, tooltip="Import CSV to project library")
            ui.Button("Export", clicked_fn=self._on_export_clicked, tooltip="Export selected CSV")
            ui.Button("Del", clicked_fn=self._on_del_clicked, tooltip="Delete selected CSV from project")
            ui.Button("Select", clicked_fn=self._on_select_clicked)
            ui.Spacer()

    def _on_import_clicked(self):
        """打开文件选择器"""
        # 创建 FilePickerDialog 时传入 current_directory
        self._import_spectral_file_picker = FilePickerDialog(
            "Import Spectral Data",
            allow_multi_selection=False,
            apply_button_label="Import",
            click_apply_handler=self._callback_import_spectral_data,  # 选择文件后的回调
            item_filter_options=[(".csv,", "USD Files")],
            current_directory=str(self.project_spectra_database_path)  # <--- 关键增加：设置默认打开路径
        )
        self._import_spectral_file_picker.show()

    def _callback_import_spectral_data(self, filename: str, dirname: str):
        """光谱数据文件选择后的回调, 将被选择的光谱数据文件导入项目文件夹
        并修改parameter文件, 向其中加入新的光谱数据文件名"""
        # 检查窗口
        if not filename or not dirname:
            if self._import_spectral_file_picker:
                self._import_spectral_file_picker.hide()
            return

        # 1. 文件路径
        src_csv_path = Path(dirname) / filename
        project_csv_path = self.project_spectra_database_path / filename

        # 2. 导入CSV
        if src_csv_path.resolve() != project_csv_path.resolve():
            shutil.copy2(src_csv_path, project_csv_path)  # 如果不是数据库中的文件，则复制

        # 3. 更新 simulation_parameters.json的光谱文件信息
        self._update_index_json(name=filename.replace(".csv", ""), file_name=filename)

        # 4. 重新加载并刷新列表
        self.database_index.clear()  # 先清空之前的字典
        self.database_index = read_spectra_file_path(self.default_parameters_path,
                                                     self.project_parameters_path)  # CSV光谱文件路径
        self._refresh_list_ui()  # 加载光谱数据到UI表格中
        print(f"Imported: {filename}")

        # 关闭窗口
        if self._import_spectral_file_picker:
            self._import_spectral_file_picker.hide()
            self._import_spectral_file_picker = None

    def _update_index_json(self, name, file_name):
        """向项目 simulation_parameters.json的光谱文件信息中 添加一条记录"""
        data = {"spectra_data_info": []}
        if self.project_parameters_path.exists():
            with open(self.project_parameters_path, 'r') as f:
                data = json.load(f)

        # 避免重复添加
        if not any(item['file'] == file_name for item in data["spectra_data_info"]):
            data["spectra_data_info"].append({"name": name, "file": file_name})
            with open(self.project_parameters_path, 'w') as f:
                json.dump(data, f, indent=4)

    def _on_export_clicked(self):
        """打开文件选择器"""
        # 创建 FilePickerDialog 时传入 current_directory
        self._export_spectral_file_picker = FilePickerDialog(
            "Export Spectral Data",
            allow_multi_selection=False,
            apply_button_label="Export",
            click_apply_handler=self._callback_export_spectral_data,  # 选择文件后的回调
            current_directory=str(self.project_spectra_database_path)  # <设置默认打开路径
        )
        self._export_spectral_file_picker.show()

    def _callback_export_spectral_data(self, filename: str, dirname: str):
        """导出当前选中的光谱数据文件, """
        # 没有选择光谱数据的提示
        if not self.current_spectrum_name:
            nm.post_notification(
                "Please select one spectral data point.",
                status=nm.NotificationStatus.INFO,
                duration=5
            )
        # 检查窗口
        if not filename or not dirname:
            if self._export_spectral_file_picker:
                self._export_spectral_file_picker.hide()
            return

        # 移动光谱数据文件
        src_path = self.database_index[self.current_spectrum_name]["csv_path"]
        target_path = Path(dirname) / filename

        # 检查后缀名是否为 .csv（忽略大小写），如果不是则强制替换/添加后缀
        if target_path.suffix.lower() != '.csv':
            target_path = target_path.with_suffix('.csv')
        shutil.copy2(src_path, target_path)

        # 关闭窗口
        if self._export_spectral_file_picker:
            self._export_spectral_file_picker.hide()
            self._export_spectral_file_picker = None

    def _on_del_clicked(self):
        """删除当前选中的光谱文件"""
        if not self.current_spectrum_name or "[Project]" not in self.current_spectrum_name:
            print("Can only delete files from Project library.")
            return

        # 1. 获取路径
        csv_path = Path(self.database_index[self.current_spectrum_name]["csv_path"])

        # 2. 从硬盘删除文件
        if csv_path.exists():
            os.remove(csv_path)

        # 3. 从 simulation_parameters.json 中移除
        self._remove_from_index_json(self.database_index[self.current_spectrum_name]["metadata"]["name"])

        # 4. 刷新
        self.database_index.clear()
        self.database_index = read_spectra_file_path(self.default_parameters_path,
                                                     self.project_parameters_path)
        self._refresh_list_ui()
        self.current_spectrum_name = None
        print(f"Deleted: {csv_path.name}")

    def _remove_from_index_json(self, name):
        """从项目 simulation_parameters.json 移除一条记录"""
        if self.project_parameters_path.exists():
            with open(self.project_parameters_path, 'r') as f:
                data = json.load(f)
            data["spectra_data_info"] = [i for i in data["spectra_data_info"] if i.get("name") != name]
            with open(self.project_parameters_path, 'w') as f:
                json.dump(data, f, indent=4)

    def _on_select_clicked(self):
        """点击 Select 按钮，解析波段并添加到表格中"""
        # 确保已经选中了列表中的某一项
        if not getattr(self, "current_spectrum_name", None):
            nm.post_notification(
                "No spectra are selected. Please click to select from the list first.",
                status=nm.NotificationStatus.WARNING,
                duration=5
            )
            return

        # 读取光谱数据
        csv_path = self.database_index[self.current_spectrum_name]["csv_path"]
        band_str = self.bands_model.as_string
        # 根据波段和光谱文件计算对应波段反射率,并格式化
        ref_str, tra_str = get_spectrum_data_for_bands(csv_path, band_str)

        # 添加到表格中
        self.data_table_model.add_row(self.current_spectrum_name, ref_str, tra_str, "0,255,0")

        # 更新simulation_parameters.json文件
        save_UI_data_to_json(self.bands_model, self.data_table_model)

        # === 窗口消失 ===
        self.visible = False

    def _build_spectral_databases_list(self):
        with ui.ScrollingFrame(width=200, style={"background_color": 0xFF222222}):
            self._list_stack = ui.VStack(spacing=5, padding=10)
            with self._list_stack:
                ui.Label("Spectral Database", style={"color": 0xFF888888}, height=20)
                ui.Line(height=2, style={"color": 0xFF444444})

                for name in self.database_index.keys():
                    ui.Button(
                        name,
                        height=30,
                        clicked_fn=lambda n=name: self._on_item_selected(n),
                        style={"text_alignment": ui.Alignment.LEFT})

    def _refresh_list_ui(self):
        """刷新左侧数据库列表 UI,实际上是一堆ui.Button按钮"""
        if not self._list_stack:  # 如果 UI 还没建好，就不刷新
            return
        self._list_stack.clear()
        with self._list_stack:
            ui.Label("Spectral Database", style={"color": 0xFF888888}, height=20)
            ui.Line(height=2, style={"color": 0xFF444444})
            for name in self.database_index.keys():
                ui.Button(name, height=30,
                          clicked_fn=lambda n=name: self._on_item_selected(n),
                          style={"text_alignment": ui.Alignment.LEFT})

    def _on_item_selected(self, name):
        """点击列表后的回调：更新数据并刷新绘图"""
        if name not in self.database_index:
            return
        self.current_spectrum_name = name
        csv_path = self.database_index[name]["csv_path"]
        refl, trans, _ = read_csv_file(csv_path)
        # 更新 Plot
        self.plot_reflectance.set_data(*refl)
        self.plot_transmittance.set_data(*trans)
        self.info_label.text = f"Selected: {name} | Bands: {len(refl)}"

    def _build_spectral_visualization(self):
        # --- 右侧：可视化区域 ---
        with ui.VStack(padding=15, spacing=10):
            self.info_label = ui.Label("Spectral Data View", height=20)

            # 使用 ZStack 将两个 Plot 叠在一起
            with ui.ZStack(height=300):
                # 1. 背景网格（底层）
                with ui.VStack():
                    for _ in range(5):
                        ui.Spacer()
                        ui.Line(height=1, style={"color": 0x22FFFFFF})

                # 2. 绘制反射率曲线 (蓝色)
                self.plot_reflectance = ui.Plot(
                    ui.Type.LINE, 0.0, 1.0,
                    style={"color": 0xFF00BFFF, "line_width": 10.0, "background_color": 0x00000000}  # 深天蓝
                )

                # 3. 绘制透射率曲线 (橙色/黄色)
                self.plot_transmittance = ui.Plot(
                    ui.Type.LINE, 0.0, 1.0,
                    style={"color": 0xFFFFBB00, "line_width": 2.0, "background_color": 0x00000000}  # 橙黄色
                )
            # X轴刻度提示
            with ui.HStack(height=20):
                ui.Rectangle(width=20, height=2, style={"background_color": 0xFF00BFFF}, alignment=ui.Alignment.CENTER)
                ui.Label(" Reflectance", width=100)
                ui.Spacer(width=20)
                ui.Rectangle(width=20, height=2, style={"background_color": 0xFFFFBB00}, alignment=ui.Alignment.CENTER)
                ui.Label(" Transmittance", width=100)

            ui.Label("Y-Axis: Reflectance (0.0 - 1.0)", style={"color": 0xFF888888, "font_size": 12})

    def _load_default_item(self):
        """默认选中第一项"""
        if self.database_index:
            first_name = list(self.database_index.keys())[0]
            self._on_item_selected(first_name)

# 在script editor中测试
# try:
#     if my_window:
#         my_window.destroy()
# except:
#     pass

# my_window = SpectralPreviewWindow()