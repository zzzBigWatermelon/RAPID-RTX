import numpy as np
import asyncio
from typing import Tuple, Dict
import laspy
import omni.ui as ui
import omni.usd
import omni.kit.commands
from pathlib import Path
from pxr import Usd, UsdGeom, Gf, Vt, Sdf, UsdShade
import os
import omni.kit.notification_manager as nm
from .window_combo_box_model import ComboBoxModel
from omni.kit.window.filepicker import FilePickerDialog
from rapid.Utility import project_validity_check
from rapid.Utility.window_components_rgb_selectors import RGBColorPickerDialog
import omni.replicator.core as rep


# 扩展程序的数据根目录
DATA_ROOT = Path(__file__).parent.parent.parent/'data'
# 点云的舞台stage路径
POINTCLOUD_STAGE_PATH = "/World/PointCloud"


class PointCloudViewerExtension:
    """
    点云数据查看器，集成在 OmnUI 窗口中。
    """

    def __init__(self):
        self._window = None
        # 所有的 UI Model 数据持久化 (UI 清空时数据不会丢)
        self._init_models()
        self._subscriptions = {}

    def _init_models(self):
        self.search_icon_path = str(DATA_ROOT / 'search.svg')
        # 初始化专门存储模型的字典
        # UI模型直接存入字典，键名与最终输出数据的键名保持一致
        self._models = {}

        # 文件选择显示路径
        self._models["image_file_path"] = ui.SimpleStringModel()

        # 点云颜色和大小UI模型
        self._models["pointcloud_rgb"] = ui.SimpleStringModel('0, 255, 0')
        self._models["pointcloud_size"] = ComboBoxModel("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")

    def show_window(self):
        """创建或显示主窗口"""
        # 检查项目文件环境完整性
        if not project_validity_check.get_current_usd_path() or not project_validity_check.quick_project_check():
            return
        # 获取参数文件路径
        self.project_result_path = str(project_validity_check.get_folder("result"))
        if self._window is None:
            self._window = ui.Window("PointCloud Viewer", width=600, height=350, visible=True)
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

                # 添加可视化点云和刷新点云大小和颜色
                with ui.HStack(height=0, spacing=0):
                    ui.Spacer()
                    ui.Button("Add PointCloud",  width=180, clicked_fn=self._on_add_pointcloud_clicked)
                    ui.Button("Refresh", clicked_fn=self._on_refresh_clicked)
                    ui.Spacer()

    def _build_file_selection_frame(self):
        """文件选择区域(CollapsableFrame)"""
        with ui.CollapsableFrame(title="File Selection", name="groupFrame", height=0, collapsed=False):
            with ui.VStack(height=0, spacing=5):
                with ui.HStack(height=0):
                    ui.Label("PointCloud File", width=120)
                    ui.StringField(model=self._models["image_file_path"], width=300)
                    ui.Button("Browse", width=80, clicked_fn=self._on_browse_clicked)

    def _build_band_settings_frame(self):
        with ui.CollapsableFrame(title="PointCloud Settings", name="groupFrame", height=0, collapsed=False):
            with ui.VStack(height=0, spacing=8, m=5):
                # 点云颜色选择
                with ui.HStack(height=22, spacing=5):
                    # 颜色选择功能
                    self._fn_color_picker()

                # 点云大小选择
                with ui.HStack(height=22, spacing=5):
                    ui.Label("PointCloud Size:", width=110)
                    # 假设 pointcloud_size 是一个 SimpleIntModel 或类似
                    ui.ComboBox(self._models["pointcloud_size"])

    def _on_browse_clicked(self):
        """浏览按钮回调：打开文件选择器"""
        # 因为在 FilePicker 中，文件夹总是可见的
        self._file_picker = FilePickerDialog(
            "Select PointCloud Data File",
            allow_multi_selection=False,
            apply_button_label="Select",
            click_apply_handler=self._fn_file_selected_callback,
            file_extension_options=[(".las", "Data Files")],
            item_filter_fn=self._fn_custom_filter,  # 使用自定义过滤函数
            current_directory=self.project_result_path
        )
        self._file_picker.show()

    def _on_add_pointcloud_clicked(self):
        '''向场景中添加可视化的点云
        '''
        # 读取文件位置，点云颜色和大小
        pointcloud_file_path = self._models["image_file_path"].as_string
        pointcloud_color = self._models["pointcloud_rgb"].as_string
        pointcloud_size = self._models["pointcloud_size"].get_current_item().as_string

        # 读取点云文件数据
        try:
            las = laspy.read(pointcloud_file_path)
        except Exception as e:
            nm.post_notification(
                f"Point cloud file failed to open.{e}",
                status=nm.NotificationStatus.WARNING,
                duration=5)
            return

        # 1. 提取位置并归一化到局部坐标系
        points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
        offset = np.mean(points, axis=0)
        local_points = points - offset

        # 2. 获取 Stage
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(POINTCLOUD_STAGE_PATH):
            usd_points = UsdGeom.Points.Define(stage, POINTCLOUD_STAGE_PATH)
        else:
            usd_points = UsdGeom.Points.Get(stage, POINTCLOUD_STAGE_PATH)

        # 3. 使用的 FromNumpy 接口赋值
        usd_points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(local_points))

        # 4. 将节点放在原始中心位置
        xformable = UsdGeom.Xformable(usd_points)
        xformable.ClearXformOpOrder()
        xform_op = xformable.AddTranslateOp()
        xform_op.Set(Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])))

        # 5. 设置宽度
        widths = np.full(len(local_points), 0.1*int(pointcloud_size), dtype=np.float32)
        usd_points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(widths))

        rgb = [float(x.strip()) for x in pointcloud_color.split(',')]
        # 获取 displayColor 属性
        color_attr = usd_points.GetDisplayColorAttr()
        # 设置统一颜色
        color_array = Vt.Vec3fArray.FromNumpy(np.array([rgb], dtype=np.float32))
        color_attr.Set(color_array)

    def _on_refresh_clicked(self):
        """刷新已有点云的颜色和大小（不改变位置）"""
        # 1. 获取 Stage 和点云 Prim
        stage = omni.usd.get_context().get_stage()

        prim = stage.GetPrimAtPath(POINTCLOUD_STAGE_PATH)
        if not prim:
            nm.post_notification(
                "PointCloud Prim not found. Please add it first.",
                status=nm.NotificationStatus.WARNING,
                duration=5)
            return

        usd_points = UsdGeom.Points(prim)  # 转换为 UsdGeom.Points 接口

        # 2. 获取当前点的数量（用于生成正确长度的宽度数组）
        points_attr = usd_points.GetPointsAttr()
        num_points = len(points_attr.Get())  # 获取点数

        # 3. 更新点云大小（Widths）
        size_str = self._models["pointcloud_size"].get_current_item().as_string
        # 宽度值（沿用原有逻辑 0.1 * size）
        new_widths = np.full(num_points, 0.1 * int(size_str), dtype=np.float32)
        widths_attr = usd_points.GetWidthsAttr()
        widths_attr.Set(Vt.FloatArray.FromNumpy(new_widths))

        # 4.点云颜色
        color_str = self._models["pointcloud_rgb"].as_string
        rgb = [float(x.strip()) for x in color_str.split(',')]
        # 创建或获取 displayColor 属性
        color_attr = usd_points.GetDisplayColorAttr()
        # 设置统一颜色
        color_array = Vt.Vec3fArray.FromNumpy(np.array([rgb], dtype=np.float32))
        color_attr.Set(color_array)

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
        return ext.lower() == ".las" or ext.lower() == ".las.pipe"

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

        # 更新路径显示的字符框label
        self._models["image_file_path"].set_value(str(full_path))

        # 关闭文件选择窗口
        self._fn_close_file_picker()

    def _fn_close_file_picker(self):
        if self._file_picker:
            self._file_picker.hide()
            self._file_picker = None

    def _fn_color_picker(self):
        ui.Label("PointCloud Color:", width=110)

        # 颜色预览方块 (Rectangle)
        color_preview = ui.Rectangle(width=20, height=20, name="color_block")
        # 颜色更新逻辑 (订阅模型变化)

        def update_color_rect(m, rect=color_preview):
            # 模型中存的是 "1.0, 0.5, 0.0" 这种格式
            parts = [float(x.strip()) for x in m.as_string.split(',')]
            if len(parts) >= 3:
                rect.style = {
                    "background_color": ui.color(parts[0], parts[1], parts[2], 1.0),
                    "border_radius": 3,
                    "border_width": 1
                }
        # 初始更新并订阅
        update_color_rect(self._models["pointcloud_rgb"])
        self._subscriptions["pointcloud_rgb_sub"] = self._models["pointcloud_rgb"].subscribe_value_changed_fn(
            lambda m: update_color_rect(m)
        )

        # 为方块绑定双击事件
        def on_rect_double_clicked(x, y, button, modifier):
            if button == 0:  # 0 代表鼠标左键
                # 弹出颜色选择器，传入模型
                RGBColorPickerDialog(self._models["pointcloud_rgb"])

        color_preview.set_mouse_double_clicked_fn(on_rect_double_clicked)
        # StringField 查看/手动修改数值
        ui.StringField(model=self._models["pointcloud_rgb"])
