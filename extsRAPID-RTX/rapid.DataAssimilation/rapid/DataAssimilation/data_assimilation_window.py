import os
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf
import omni.usd
from PIL import Image
import numpy as np
import time
from pathlib import Path
from isaacsim.gui.components.style import get_style
from omni.kit.window.filepicker import FilePickerDialog
# from omni.example.ui_scatter_tool.style import scatter_window_style
from omni.example.ui_scatter_tool.combo_box_model import ComboBoxModel
from rapid.Utility import project_validity_check

LABEL_HEIGHT = 24
LABEL_WIDTH = 18
SPACING = 8

# 玉米stage路径
STAGE_YUMI = ['/YUMI/Xform', '/YUMI/Xform_01', '/YUMI/Xform_02', '/YUMI/Xform_03', '/YUMI/Xform_04', '/YUMI/Xform_05',
              '/YUMI/Xform_06', '/YUMI/Xform_07', '/YUMI/Xform_08', '/YUMI/Xform_09', '/YUMI/Xform_10', '/YUMI/Xform_11',
              '/YUMI/Xform_12', '/YUMI/Xform_13', '/YUMI/Xform_14', '/YUMI/Xform_15', '/YUMI/Xform_16', '/YUMI/Xform_17',
              '/YUMI/Xform_18', '/YUMI/Xform_19', '/YUMI/Xform_20', '/YUMI/Xform_21', '/YUMI/Xform_22', '/YUMI/Xform_23',
              '/YUMI/Xform_24']

DEFAULT_IMAGE = str(Path(__file__).parent.parent.parent/'data'/'No Data.png')


class DataAssimilationWindow(ui.Window):

    def __init__(self, title: str, delegate=None, **kwargs):

        # 设定窗口高度
        self.__label_height = LABEL_HEIGHT

        super().__init__(title, **kwargs)

        # 将日期列表提取出来，方便后续通过索引获取文本
        self.date_items = [
            "June 1", 'June 10', "June 20",
            "July 1", 'July 10', "July 20",
            "August 1", 'August 10', "August 20",
            "September 1", 'September 10', "September 20"
        ]

        # 定义日期与缩放比例的映射关系 (模拟生长，逐渐变大)
        # 假设 June 1 是 1.0倍，后面每过一个阶段增加 0.002
        self.date_to_scale_map = {}
        base_scale = 0.01
        for i, date_str in enumerate(self.date_items):
            scale_val = base_scale + (i * 0.002)
            self.date_to_scale_map[date_str] = Gf.Vec3d(scale_val, scale_val, scale_val)

        # Models
        self._assimilation_variable_type = ComboBoxModel("CHM", "LAI", "LiDAR data")   # 数据同化变量
        self._simulated_data_source = ComboBoxModel("Terrestrial LiDAR", "Tower bas", 'Airborne LiDAR', "UAV")   # 数据同化变量
        self._growth_date = ComboBoxModel(*self.date_items)   # 数据同化变量
        self._file_picker = None   # 实测数据文件选择

        # 监听 Growth Date 的变化
        self._growth_date.get_item_value_model(None, 0).add_value_changed_fn(self._on_growth_date_changed)

        # 用于存储 UI 图片控件的引用
        self._simulation_image_widget = None
        self._measured_image_widget = None
        self._assimulation_result_image_widget = None

        # Apply the style to all the widgets of this window
        # self.frame.style = scatter_window_style
        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_frame_data_assimilation)

    def destroy(self):
        # It will destroy all the children
        super().destroy()

        if self._file_picker:
            self._file_picker.hide()
            self._file_picker = None

    def _build_frame_data_assimilation(self):
        with ui.VStack(height=0, spacing=SPACING):
            ui.Label("Data assimilation", height=20, style={"font_size": 18, "color": 0xFFFFFFFF},
                     alignment=ui.Alignment.CENTER)

            # 同化模拟选择按钮(一个选择同化变量、一个选择模拟数据源,一个选择同化日期)
            self._build_frame_select_simulation_model()

            # 模拟和实测数据导入
            self._build_frame_select_data()

            # 放置三个数据视图
            self._build_frame_data_view()

            # 数据同化按钮
            ui.Button("Data Assimilation", width=1000, height=24, clicked_fn=self._execute_assimilation,
                      tooltip="Data Assimilation", alignment=ui.Alignment.CENTER)

    def _build_frame_select_simulation_model(self):
        with ui.HStack(spacing=20):
            ui.Label("Assimilation Variable:", width=LABEL_WIDTH, style=get_style())
            ui.ComboBox(self._assimilation_variable_type)
            ui.Label("Simulated Data:", width=LABEL_WIDTH, style=get_style())
            ui.ComboBox(self._simulated_data_source)
            ui.Label("Growth Date:", width=LABEL_WIDTH, style=get_style())
            ui.ComboBox(self._growth_date)

    def _build_frame_select_data(self):
        # 模拟和实际观测数据选择窗口
        with ui.HStack(height=24):
            # 模拟数据选择
            ui.Label('Simulated Data:', width=100, alignment=ui.Alignment.LEFT_CENTER)
            # 路径显示框 (StringField)
            simulation_str_field = ui.StringField(width=ui.Fraction(1), alignment=ui.Alignment.LEFT_CENTER).model
            simulation_str_field.set_value("C:")
            # 文件选择按钮
            self.add_folder_picker_icon(simulation_str_field, "simulation")

            # 实测数据选择
            ui.Label('Select Actual Measured Data:', width=100, alignment=ui.Alignment.LEFT_CENTER)
            # 路径显示框 (StringField)
            measured_str_field = ui.StringField(width=ui.Fraction(1), alignment=ui.Alignment.LEFT_CENTER).model
            measured_str_field.set_value("C:")
            # 文件选择按钮
            self.add_folder_picker_icon(measured_str_field, "measured")

    def add_folder_picker_icon(self, str_field, widget_key):
        """创建图标按钮并处理文件选择器的显示"""
        # 文件选择函数
        def open_file_picker():
            # 内部回调：当在弹窗中点击确定时
            def on_selected(filename, path):
                # 路径拼接
                if not filename:
                    full_path = path
                elif path.endswith("/") or filename.startswith("/"):
                    full_path = path + filename if not (path.endswith("/") and filename.startswith("/")) else path + filename[1:]
                else:
                    full_path = f"{path}/{filename}"

                # 将路径填入输入框
                str_field.set_value(full_path)

                # 获取最新的控件引用
                current_image_widget = None
                if widget_key == "simulation":
                    current_image_widget = self._simulation_image_widget
                elif widget_key == "measured":
                    current_image_widget = self._measured_image_widget
                # 更新数据图片
                if current_image_widget and full_path:
                    current_image_widget.source_url = full_path

                # 隐藏文件选择窗口
                self.file_picker.hide()

            def on_canceled(a, b):
                self.file_picker.hide()

            # 获取当前打开的usd文件路径
            assimilation_path = project_validity_check.get_folder("data_assimilation")
            # 创建文件选择器对话框
            self.file_picker = FilePickerDialog(
                "Select Actual Measured Data",  # 文件选择器对话框名称
                allow_multi_selection=False,
                apply_button_label='Select',
                click_apply_handler=lambda a, b: on_selected(a, b),
                click_cancel_handler=lambda a, b: on_canceled(a, b),
                current_directory=assimilation_path  # 默认打开的路径
            )
            self.file_picker.show()

        # 这里用一个简单图标
        ui.Button(
            "folder.svg",
            width=30,
            height=24,
            clicked_fn=open_file_picker,
            tooltip="Select File"
        )

    def _build_frame_data_view(self):
        # 定义框的样式
        frame_style = {
            "background_color": 0xFF222222,
            "border_color": 0xFF777777,
            "border_width": 1.0,
            "border_radius": 5.0
        }

        # 图片下的名字列表
        image_name = ['Simulation Data', 'Actual Measured Data', 'Analysis']
        with ui.HStack(spacing=30, width=1000):
            for i in range(3):
                with ui.VStack(spacing=8):
                    # --- 核心：框 ---
                    with ui.Frame(style=frame_style, height=230):
                        # 创建图片控件
                        img = ui.Image(
                            DEFAULT_IMAGE,
                            alignment=ui.Alignment.CENTER)

                        # 捕获第一张模拟数据图的引用
                        if i == 0:
                            self._simulation_image_widget = img
                        # 捕获第二张实测数据图的引用
                        if i == 1:
                            self._measured_image_widget = img
                        if i == 2:
                            self._assimulation_result_image_widget = img

                    # 框下方的描述文字
                    ui.Label(image_name[i], height=15, alignment=ui.Alignment.CENTER)

    def _on_growth_date_changed(self, model):
        # model 是 SimpleIntModel，返回的是ComboBoxModel中内容的序号
        index = model.as_int

        if 0 <= index < len(self.date_items):
            selected_date = self.date_items[index]  # 日期
            target_scale = self.date_to_scale_map.get(selected_date)  # scale数值

            if target_scale:
                for root_path in STAGE_YUMI:
                    self._update_prim_scale(root_path, target_scale)

    def _update_prim_scale(self, root_path, scale_vec):
        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        if not stage:
            return

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return

        # 遍历所有子类xform
        children = root_prim.GetChildren()

        for prim in children:
            # 过滤：确保是一个可变换的几何体 (Xform, Mesh 等)，排除材质或Scope等
            if prim.IsA(UsdGeom.Xformable):
                self._apply_scale_to_single_prim(prim, scale_vec)

    def _apply_scale_to_single_prim(self, prim, scale_vec):
        """
        辅助函数：对单个 Prim 执行具体的缩放操作
        """
        xformable = UsdGeom.Xformable(prim)
        scale_op = None
        # 查找现有的 Scale 操作
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        # 如果没有，添加一个新的
        if not scale_op:
            scale_op = xformable.AddScaleOp()
        # 设置值
        scale_op.Set(scale_vec)

    def _execute_assimilation(self):
        """读取图1和图2,计算平均值,保存并显示在图3位置"""
        print("[Rapid.DataAssimilation] Executing Data Assimilation...")

        # 检查图片引用是否存在
        if not (self._simulation_image_widget and self._measured_image_widget and self._assimulation_result_image_widget):
            print("[Error] Image widgets not initialized.")
            return

        # 获取路径
        path1 = self._simulation_image_widget.source_url
        path2 = self._measured_image_widget.source_url

        # 检查文件是否存在
        if not (os.path.exists(path1) and os.path.exists(path2)):
            print(f"[Error] Source images not found.\nPath1: {path1}\nPath2: {path2}")
            return

        try:
            # 加载图片并转换为 RGB
            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')

            # 统一尺寸 (以第一张图为基准，调整第二张图大小)
            if img1.size != img2.size:
                print(f"[Info] Resizing Image 2 from {img2.size} to {img1.size}")
                img2 = img2.resize(img1.size)

            # 转换为 Numpy 数组进行计算
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)

            # 计算像素平均值
            avg_arr = (arr1 + arr2) / 2.0

            # 转回 uint8 图像
            result_img = Image.fromarray(np.uint8(avg_arr))

            # 在保存前，检查文件夹是否存在
            assimilation_path = project_validity_check.get_folder("data_assimilation")
            assimilation_result_path = os.path.join(assimilation_path, 'Assimilation Result', 'CHM')
            if not os.path.exists(assimilation_result_path):
                os.makedirs(assimilation_result_path, exist_ok=True)

            # 保存结果图片
            # 使用时间戳防止文件名冲突和缓存问题
            filename = f"assimilation_result_{int(time.time())}.png"
            save_path = os.path.join(assimilation_result_path, filename)

            result_img.save(save_path)
            print(f"[Success] Result saved to: {save_path}")

            # 8. 更新第三张图片的显示
            self._assimulation_result_image_widget.source_url = save_path

            # model 是 SimpleIntModel，返回的是ComboBoxModel中内容的序号
            s = self._growth_date.get_current_item().as_string

            index = self.date_items.index(s)
            if 0 <= index < len(self.date_items):
                selected_date = self.date_items[index]  # 日期
                target_scale = self.date_to_scale_map.get(selected_date)  # scale数值

                if target_scale:
                    for root_path in STAGE_YUMI:
                        self._update_prim_scale(root_path, target_scale*1.3)

        except Exception as e:
            print(f"[Error] Assimilation failed: {e}")
            import traceback
            traceback.print_exc()
