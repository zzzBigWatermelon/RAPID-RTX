import numpy as np
from PIL import Image as PILImage
from pathlib import Path
import omni.kit.notification_manager as nm
import omni.ui as ui
import os
import spectral.io.envi as envi
from .window_combo_box_model import ComboBoxModel
from omni.kit.window.filepicker import FilePickerDialog
from rapid.Utility import project_validity_check



class MyImageViewerExtension:
    """
    ENVI 高光谱图像查看器，集成在 OmnUI 窗口中。
    """

    def __init__(self):
        self._window = None

    def _init_models(self):
        # 初始化专门存储模型的字典
        # UI模型直接存入字典，键名与最终输出数据的键名保持一致
        self._models = {}

        # 图像文件选择显示路径
        self._models["image_file_path"] = ui.SimpleStringModel()
        self._wavelengths = []   # 存储从图像头文件读取的波长列表（浮点数）

        # 波段选择模型的默认值（R, G, B 波段索引）
        self._models["band_r_model"] = ComboBoxModel("Select Band")
        self._models["band_g_model"] = ComboBoxModel("Select Band")
        self._models["band_b_model"] = ComboBoxModel("Select Band")

    def show_window(self):
        """创建或显示主窗口"""
        # 所有的 UI Model 数据持久化 (UI 清空时数据不会丢),只在第一次打开窗口时创建
        if self._window is None:
            self._init_models()
        # 检查项目文件环境完整性
        if not project_validity_check.get_current_usd_path() or not project_validity_check.quick_project_check():
            return
        # 获取参数文件路径
        self.project_result_path = str(project_validity_check.get_folder("result"))
        if self._window is None:
            self._window = ui.Window("Image Viewer", width=700, height=750, visible=True)
            with self._window.frame:
                self._build_fn()
        else:
            self._window.visible = True

    def _build_fn(self):
        """主布局构建函数，参考你提供的风格"""
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8, height=0):
                self._build_file_selection_frame()
                self._build_band_settings_frame()
                self._build_image_display_frame()

    def _build_file_selection_frame(self):
        """文件选择区域(CollapsableFrame)"""
        with ui.CollapsableFrame(title="File Selection", name="groupFrame", height=0, collapsed=False):
            with ui.VStack(height=0, spacing=5):
                with ui.HStack(height=0):
                    ui.Label("ENVI Header File", width=120)
                    ui.StringField(model=self._models["image_file_path"], width=300)
                    ui.Button("Browse", width=80, clicked_fn=self._on_browse_clicked)

    def _build_band_settings_frame(self):
        with ui.CollapsableFrame(title="RGB Band Selection", name="groupFrame", height=0, collapsed=False):
            with ui.VStack(height=0, spacing=5):
                with ui.HStack():
                    ui.Label("Red Band:", width=100)
                    ui.ComboBox(self._models["band_r_model"])

                    ui.Label("Green Band:", width=100)
                    ui.ComboBox(self._models["band_g_model"])

                    ui.Label("Blue Band:", width=100)
                    ui.ComboBox(self._models["band_b_model"])

    def _build_image_display_frame(self):
        """图像显示区域"""
        with ui.CollapsableFrame(title="Image Display", name="groupFrame", height=0, collapsed=False):
            with ui.VStack(height=0, spacing=5):
                # 用一个固定高度的 VStack 来承载图像，不使用 ui.Frame
                self._image_container = ui.VStack(height=500, spacing=0)
                with self._image_container:
                    # 初始占位标签
                    self._image_placeholder = ui.Label("No image loaded. Click 'Browse' to select an ENVI file.",
                                                       alignment=ui.Alignment.CENTER, height=500)
                with ui.HStack():
                    ui.Spacer()
                    ui.Button("Show Image", width=150, clicked_fn=self._show_image)
                    ui.Spacer()

    def _on_browse_clicked(self):
        """浏览按钮回调：打开文件选择器"""
        # 因为在 FilePicker 中，文件夹总是可见的
        self._file_picker = FilePickerDialog(
            "Select File or Folder",
            allow_multi_selection=False,
            apply_button_label="Select",
            click_apply_handler=self._fn_file_selected_callback,
            file_extension_options=[(".hdr", "image Files")],
            item_filter_fn=self._fn_custom_filter,  # 使用自定义过滤函数
            current_directory=self.project_result_path
        )
        self._file_picker.show()

    def _fn_custom_filter(self, item):
        """
        定义一个自定义的过滤函数
        根据文件扩展名决定是否在列表中显示。
        True: 显示该文件/文件夹
        False: 隐藏该文件/文件夹
        """
        if not item:
            return False
        if item.is_folder:
            return True
        # 检查文件扩展名是否为 .las，注意大小写不敏感
        _, ext = os.path.splitext(item.path)
        return ext.lower() == ".hdr"

    def _fn_file_selected_callback(self, filename: str, dirname: str):
        """
        选中后的回调函数
        参数:
            filename (自动传参): 用户选中的文件名（如果是选文件夹，这通常是空的或者是文件夹名）
            dirname (自动传参): 目录路径
        """
        # 文件路径检查
        if filename:
            full_path = Path(dirname) / filename
        else:
            full_path = dirname
            # 如果不存在则直接返回
            nm.post_notification(
                "Please select an image file instead of a folder.",
                status=nm.NotificationStatus.WARNING,
                duration=5)

        # 更新图像文件路径UI
        self._models["image_file_path"].set_value(str(full_path))

        # 读取头文件
        img = envi.open(full_path)
        wavelengths_str = img.metadata.get('wavelength')
        self._wavelengths = [float(w) for w in wavelengths_str]
        # 更新波段选择UI
        self._update_band_combos(self._wavelengths)

        # 关闭文件选择窗口
        self._close_picker()

    def _update_band_combos(self, wavelengths):
        """当选中hdr图像文件后,更新三个 ComboBox 的选项列表"""
        if not wavelengths:
            items = ["No wavelength data"]
        else:
            # 格式化为 "Band1 (400.5 nm)" 样式
            items = [f"Band{i+1} ({w:.2f} nm)" for i, w in enumerate(wavelengths)]
        # ComboBoxModel 有 set_items 方法
        self._models["band_r_model"].set_items(items)
        self._models["band_g_model"].set_items(items)
        self._models["band_b_model"].set_items(items)

        # 设置默认选中索引，通过combox默认索引_default设定数值
        # R 默认第一个
        self._models["band_r_model"]._default.as_int = 0
        # G 默认第二个（如果存在），否则第一个
        if len(items) > 1:
            self._models["band_g_model"]._default.as_int = 1
        else:
            self._models["band_g_model"]._default.as_int = 0

        # B 默认第三个（如果存在），否则第一个
        if len(items) > 2:
            self._models["band_b_model"]._default.as_int = 2
        else:
            self._models["band_b_model"]._default.as_int = 0

    def _close_picker(self):
        if self._file_picker:
            self._file_picker.hide()
            self._file_picker = None

    def _load_and_display(self, file_path: str):
        """加载 ENVI 文件并更新显示"""
        # 读取图像数据
        img = envi.open(file_path)
        data_array = img.load()  # 形状: (rows, cols, bands)

        # 获取选定波段的index
        band_r_index = self._models["band_r_model"]._default.as_int
        band_g_index = self._models["band_g_model"]._default.as_int
        band_b_index = self._models["band_b_model"]._default.as_int

        # 提取 RGB 波段数据
        # np.squeeze的作用，去掉多余的维度(300,300,1) → (300,300)
        r_band_data = np.squeeze(data_array[:, :, band_r_index])
        g_band_data = np.squeeze(data_array[:, :, band_g_index])
        b_band_data = np.squeeze(data_array[:, :, band_b_index])

        # 归一化到 0-255
        def normalize(band):
            band_min, band_max = band.min(), band.max()
            if band_max - band_min == 0:
                return np.zeros_like(band, dtype=np.uint8)
            normalized = (band - band_min) / (band_max - band_min) * 255
            return normalized.astype(np.uint8)

        # 合成一张图像
        rgb_img_data = np.stack([normalize(r_band_data), normalize(g_band_data), normalize(b_band_data)], axis=-1)
        pil_img = PILImage.fromarray(rgb_img_data)

        # 保存到临时文件
        # 获取对应的波段数值
        w_r = self._wavelengths[band_r_index]
        w_g = self._wavelengths[band_g_index]
        w_b = self._wavelengths[band_b_index]
        image_path = Path(file_path).parent / f"view_{w_r:.1f}_{w_g:.1f}_{w_b:.1f}.png"
        pil_img.save(image_path)

        # 更新显示
        self._display_final_image(image_path, pil_img.width, pil_img.height)

    def _show_image(self):
        """手动刷新显示（当波段索引改变时）"""
        # 图像路径
        file_path = self._models["image_file_path"].as_string

        if file_path:
            # 如果还没加载过，先加载
            self._load_and_display(file_path)
        else:
            # 如果不存在则直接返回
            nm.post_notification(
                "Please select an image file.",
                status=nm.NotificationStatus.WARNING,
                duration=5)
            return

    def _display_final_image(self, image_path, w, h):
        """按照官方文档风格进行 UI 构建"""
        self._image_container.clear()

        # 确保 image_path 是字符串而不是 Path 对象
        # .as_posix() 会将 WindowsPath 转换为 'C:/Users/...' 这种字符串
        if hasattr(image_path, "as_posix"):
            image_string = image_path.as_posix()
        else:
            image_string = str(image_path)

        with self._image_container:
            # 参考官方文档：直接传入路径字符串
            # 使用 PRESERVE_ASPECT_FIT 确保图片不失真
            ui.Image(
                image_string,
                width=ui.Percent(100),
                height=ui.Percent(100),
                fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                alignment=ui.Alignment.CENTER
            )

    def _show_error_message(self, message: str):
        """在图像显示区域显示错误信息"""
        self._image_container.clear()
        with self._image_container:
            ui.Label(message, alignment=ui.Alignment.CENTER, height=500, word_wrap=True)
