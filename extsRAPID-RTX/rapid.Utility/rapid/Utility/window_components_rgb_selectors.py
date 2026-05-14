import omni.ui as ui
import asyncio
import omni.kit.app
from isaacsim.gui.components.style import get_style
from pathlib import Path

# 定义路径
DATA_ROOT = Path(__file__).parent.parent.parent/'data'
# 样式常量
COLOR_R = 0xFF5353FF
COLOR_G = 0xFF76D321
COLOR_B = 0xFF21A0FF
# 加大后的标签尺寸
RGB_LABEL_WIDTH = 35
RGB_FIELD_HEIGHT = 30
TITLE_WIDTH = 100      # 左侧统一标题宽度


class RGBColorPickerDialog(ui.Window):
    def __init__(self, target_model, title="Color Picker"):
        super().__init__(title, width=500, height=280, flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)

        self.target_model = target_model  # 外部表格的 StringModel
        self._ignore_callback = False
        # 放大镜图标的位置
        self.search_icon_path = str(DATA_ROOT / "search.svg")
        # 1. 解析初始颜色 (从 "0.5, 0.5, 0.5" 字符串解析)
        init_rgb = [1.0, 1.0, 1.0]
        try:
            # 兼容 float 格式
            parts = [float(x.strip()) for x in self.target_model.as_string.split(',')]
            if len(parts) >= 3:
                init_rgb = parts[:3]
        except:
            pass

        with self.frame:
            with ui.VStack(padding=15, spacing=15):

                # --- 第一部分：颜色选择 + RGB 数值 (内部互相同步) ---
                with ui.HStack(spacing=20, height=140):
                    # 左侧颜色盘
                    with ui.VStack(width=130):
                        self.color_widget = ui.ColorWidget(*init_rgb, 1.0, width=130, height=130)
                        self.color_model = self.color_widget.model

                    # 右侧 RGB 拖拽框
                    with ui.VStack(spacing=8):
                        ui.Label("Spectral RGB Channels", height=20, style={"font_size": 14, "color": 0xFF888888})
                        self.temp_rgb_models = self._build_rgb_inputs(init_rgb)
                        ui.Spacer()

                # --- 第二部分：Color Image 行 ---
                with ui.HStack(height=RGB_FIELD_HEIGHT, spacing=10):
                    ui.Label("Color Image", width=TITLE_WIDTH)
                    self.image_path_field = ui.StringField(name="StringField")
                    ui.Button(
                        name="IconButton", width=30, height=RGB_FIELD_HEIGHT,
                        image_url=self.search_icon_path, tooltip="Select Image File",
                        alignment=ui.Alignment.CENTER,
                        clicked_fn=lambda: print("Open File Browser")
                    )

                ui.Spacer()

                # --- 第三部分：确认按钮 (此处才执行同步) ---
                with ui.HStack(height=35):
                    ui.Spacer()  # 左侧弹簧，把按钮向右推

                    ui.Button(
                        "Confirm",
                        width=120,   # 给按钮一个固定宽度
                        height=35,
                        clicked_fn=self._on_confirm_clicked,
                        style=get_style()
                    )
                    ui.Spacer()  # 右侧弹簧，把按钮向左推

        # 内部双向绑定：颜色盘 <-> RGB输入框
        self._setup_internal_callbacks()

    def _build_rgb_inputs(self, default_val):
        models = []
        labels = [("R", COLOR_R), ("G", COLOR_G), ("B", COLOR_B)]
        for i in range(3):
            with ui.HStack(height=RGB_FIELD_HEIGHT):
                with ui.ZStack(width=RGB_LABEL_WIDTH):
                    ui.Rectangle(style={"background_color": labels[i][1], "border_radius": 3})
                    ui.Label(labels[i][0], alignment=ui.Alignment.CENTER, style={"color": 0xFFFFFFFF, "font_weight": "bold"})
                ui.Spacer(width=8)
                drag = ui.FloatDrag(min=0.0, max=1.0, step=0.001)
                drag.model.set_value(default_val[i])
                models.append(drag.model)
        return models

    def _setup_internal_callbacks(self):
        self._subs = []
        # 内部同步：颜色盘变化时更新拖拽框
        for child in self.color_model.get_item_children():
            m = self.color_model.get_item_value_model(child)
            self._subs.append(m.subscribe_value_changed_fn(self._sync_widget_to_drags))
        # 内部同步：拖拽框变化时更新颜色盘
        for m in self.temp_rgb_models:
            self._subs.append(m.subscribe_value_changed_fn(self._sync_drags_to_widget))

    def _sync_widget_to_drags(self, _):
        if self._ignore_callback:
            return
        self._ignore_callback = True
        children = self.color_model.get_item_children()
        for i in range(3):
            val = self.color_model.get_item_value_model(children[i]).as_float
            self.temp_rgb_models[i].set_value(val)
        self._ignore_callback = False

    def _sync_drags_to_widget(self, _):
        if self._ignore_callback:
            return
        self._ignore_callback = True
        children = self.color_model.get_item_children()
        for i in range(3):
            self.color_model.get_item_value_model(children[i]).set_value(self.temp_rgb_models[i].as_float)
        self._ignore_callback = False

    def _on_confirm_clicked(self):
        """核心逻辑：点击确认时，将值写回表格的 StringModel"""
        r = round(self.temp_rgb_models[0].as_float, 3)
        g = round(self.temp_rgb_models[1].as_float, 3)
        b = round(self.temp_rgb_models[2].as_float, 3)

        # 拼接为字符串，存入表格 model
        result_str = f"{r}, {g}, {b}"
        self.target_model.set_value(result_str)

        # 关闭窗口：使用异步延迟销毁
        async def defer_destroy():
            # 等待下一帧更新
            await omni.kit.app.get_app().next_update_async()
            # 现在安全了，销毁窗口
            self.destroy()

        # 启动异步任务
        asyncio.ensure_future(defer_destroy())

    def destroy(self):
        self._subs = []
        super().destroy()