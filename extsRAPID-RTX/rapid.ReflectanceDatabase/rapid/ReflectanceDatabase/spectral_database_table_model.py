import omni.ui as ui
import asyncio
import omni.kit.app
from isaacsim.gui.components.style import get_style
import os
from pathlib import Path


# 颜色定义
COLOR_NORMAL = 0xFF222222      # 普通背景 (深灰)
COLOR_SELECTED = 0xFFD06020    # 选中高亮 (亮蓝色)
COLOR_BORDER = 0xFF444444
# 样式常量
COLOR_R = 0xFF5353FF
COLOR_G = 0xFF76D321
COLOR_B = 0xFF21A0FF
# 加大后的标签尺寸
RGB_LABEL_WIDTH = 35
RGB_FIELD_HEIGHT = 30
TITLE_WIDTH = 100      # 左侧统一标题宽度
DATA_ROOT = Path(__file__).parent.parent.parent/'data'


# 表格UI的模型
class TableItem(ui.AbstractItem):
    """
    参考omni.example.ui.tree_view.py
    根据列的类型自动选择 StringModel 还是 FloatModel
    """
    def __init__(self, param_name, ref, tra, display_color="1.0, 1.0, 1.0"):
        super().__init__()
        # 第1列: String
        self.name_model = ui.SimpleStringModel(str(param_name))
        # 第2列: String
        self.ref_value_model = ui.SimpleStringModel(str(ref))
        # 第3列: String
        self.tra_value_model = ui.SimpleStringModel(str(tra))
        # 第3列: String
        self.display_color_model = ui.SimpleStringModel(str(display_color))

        # 每一行都有一个布尔模型，控制是否高亮
        self.is_selected_model = ui.SimpleBoolModel(False)


class TableModel(ui.AbstractItemModel):
    '''参考omni.example.ui.tree_view.py
    '''
    def __init__(self, headers, data):
        super().__init__()
        # headers应该包含四列内容，与TableItem相同
        self.headers = headers
        self._items = [TableItem(*row) for row in data]

    def get_item_children(self, item):
        return self._items if item is None else []

    def get_item_value_model_count(self, item):
        return len(self.headers)

    def get_item_value_model(self, item, column_id):
        if item is None:
            return None
        # 根据列 ID 返回对应的 Model
        if column_id == 0:
            return item.name_model
        elif column_id == 1:
            return item.ref_value_model
        elif column_id == 2:
            return item.tra_value_model
        elif column_id == 3:
            return item.display_color_model
        return None

    def reset_data(self, new_data_list):
        """清空旧数据并加载新数据"""
        self._items.clear()
        for row in new_data_list:
            if len(row) >= 4:  # 确保数据长度足够
                self._items.append(TableItem(row[0], row[1], row[2], row[3]))
            elif len(row) == 3:  # 兼容旧的3列格式数据
                self._items.append(TableItem(row[0], row[1], row[2], "1,1,1"))
        # 通知 UI 全局重绘
        self._item_changed(None)

    def add_row(self, name, ref, tra, display_color="1,1,1"):
        '''新增行'''
        item = TableItem(name, ref, tra, display_color)
        self._items.append(item)
        self._item_changed(None)

    def remove_items(self, items_to_delete):
        '''删除行'''
        dirty = False
        for item in items_to_delete:
            if item in self._items:
                self._items.remove(item)
                dirty = True
        if dirty:
            self._item_changed(None)

    # 处理选中逻辑
    def set_single_selection(self, target_item):
        """将目标设为选中，其他全部取消选中"""
        for item in self._items:
            # 如果是目标，设为 True；否则设为 False
            is_target = (item == target_item)
            if item.is_selected_model.as_bool != is_target:
                item.is_selected_model.as_bool = is_target


class TableDelegate(ui.AbstractItemDelegate):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self._subscriptions = {}
        self.label_height = 24

    def build_branch(self, model, item, column_id, level, expanded):
        pass

    def build_header(self, column_id):
        title = self._model.headers[column_id] if self._model and column_id < len(self._model.headers) else ""
        with ui.ZStack(height=self.label_height):
            ui.Rectangle(style={"background_color": 0xFF3A3A3A, "border_color": 0xFF555555, "border_width": 1})
            ui.Label(title, alignment=ui.Alignment.CENTER, style={"color": 0xFFAAAAAA, "font_size": 14})

    def build_widget(self, model, item, column_id, level, expanded):
        if item is None:
            return

        value_model = model.get_item_value_model(item, column_id)

        # 容器
        stack = ui.ZStack(height=self.label_height)
        with stack:
            # 1. 默认背景 (深灰) - 永远显示
            ui.Rectangle(style={"background_color": COLOR_NORMAL, "border_color": COLOR_BORDER, "border_width": 1})

            # 2. 高亮背景 (亮蓝) - 默认隐藏，覆盖在普通背景上
            # 这里我们不依赖 redraw，而是让它一直存在，只控制 visible
            highlight_rect = ui.Rectangle(
                style={"background_color": COLOR_SELECTED},
                visible=item.is_selected_model.as_bool  # 初始状态
            )

            with ui.HStack(spacing=5):
                ui.Spacer(width=5)

                # 第四列：显示颜色方块
                if column_id == 3:
                    color_preview = ui.Rectangle(width=16, height=16)

                    def update_color_rect(m, rect=color_preview):
                        try:
                            parts = [float(x.strip()) for x in m.as_string.split(',')]
                            if len(parts) >= 3:
                                rect.style = {"background_color": ui.color(parts[0], parts[1], parts[2], 1.0), "border_radius": 3}
                        except Exception:
                            print(f"Color parse error: {e}")
                            rect.style = {"background_color": 0xFF888888}

                    update_color_rect(value_model)
                    self._subscriptions[id(color_preview)] = value_model.subscribe_value_changed_fn(update_color_rect)

                # 文本层
                label = ui.Label(value_model.as_string, alignment=ui.Alignment.LEFT_CENTER)

                # 订阅文本变化
                def update_label(m, l=label):
                    l.text = m.as_string
                self._subscriptions[f"{id(item)}_{column_id}_label"] = value_model.subscribe_value_changed_fn(update_label)

            # 4. 透明点击捕获层 (Hit Cover)
            hit_cover = ui.Rectangle(style={"background_color": 0x01000000}, visible=True)

            # 5. 编辑框 (最上层)
            field = ui.StringField(value_model, visible=False, style={"background_color": 0xFF000000})

        # --- 绑定双击事件 ---
        # 我们需要把 column_id 传给 _start_edit
        hit_cover.set_mouse_double_clicked_fn(
            lambda x, y, b, m, f=field, l=label, c=column_id, mod=value_model: 
            self._start_edit(b, f, l, c, mod)
        )

        # 单击选中逻辑
        hit_cover.set_mouse_pressed_fn(lambda x, y, b, m, i=item: self._on_single_click(b, i))

        # 订阅高亮
        sub_id_vis = id(highlight_rect)
        self._subscriptions[sub_id_vis] = item.is_selected_model.subscribe_value_changed_fn(
            lambda m, w=highlight_rect: setattr(w, "visible", m.as_bool)
        )

    def _on_single_click(self, button, item):
        if button == 0:  # 左键
            # 这会触发 subscribe_value_changed_fn，瞬间切换显隐
            self._model.set_single_selection(item)

    def _start_edit(self, button, field, label, column_id, model):
        if button != 0:
            return

        if column_id == 3:
            # --- 第四列：弹出专业颜色选择器 ---
            RGBColorPickerDialog(model)
        else:
            # --- 前三列：保持原有的文本输入模式 ---
            field.visible = True

            async def focus_with_delay():
                await omni.kit.app.get_app().next_update_async()
                field.focus_keyboard()
            asyncio.ensure_future(focus_with_delay())

            sub_id = id(field)
            self._subscriptions[sub_id] = field.model.subscribe_end_edit_fn(
                lambda m, f=field, l=label, sid=sub_id: self._end_edit(m, f, l, sid)
            )

    def _end_edit(self, model, field, label, sub_id):
        field.visible = False
        label.text = model.as_string
        if sub_id in self._subscriptions:
            del self._subscriptions[sub_id]


class RGBColorPickerDialog(ui.Window):
    def __init__(self, target_model, title="Color Picker"):
        super().__init__(title, width=500, height=280, flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)

        self.target_model = target_model  # 外部表格的 StringModel
        self._ignore_callback = False 
        self.search_icon_path = str(DATA_ROOT / 'search.svg')
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