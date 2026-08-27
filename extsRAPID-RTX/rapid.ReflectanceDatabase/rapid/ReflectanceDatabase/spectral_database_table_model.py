import omni.ui as ui
import asyncio
import omni.kit.app
from isaacsim.gui.components.style import get_style
from pathlib import Path


import omni.ui as ui
from pathlib import Path


# ============================================================
# Colors
# ============================================================

COLOR_NORMAL = 0xFF222222
COLOR_BORDER = 0xFF444444

COLOR_HEADER = 0xFF3A3A3A
COLOR_HEADER_BORDER = 0xFF555555
COLOR_HEADER_TEXT = 0xFFAAAAAA

# 当前选中 Cell 的编辑背景
COLOR_EDITING = 0xFF151515

# RGB
COLOR_R = 0xFF5353FF
COLOR_G = 0xFF76D321
COLOR_B = 0xFF21A0FF

RGB_LABEL_WIDTH = 35
RGB_FIELD_HEIGHT = 30
TITLE_WIDTH = 100

DATA_ROOT = Path(__file__).parent.parent.parent / "data"


# ============================================================
# Table Item
# ============================================================

class TableItem(ui.AbstractItem):
    """
    表格中的一行。

    一个 TableItem 对应 TreeView 中的一行。
    """

    def __init__(
        self,
        param_name,
        ref,
        tra,
        display_color="1.0, 1.0, 1.0"
    ):
        super().__init__()

        # Column 0
        self.name_model = ui.SimpleStringModel(
            str(param_name)
        )

        # Column 1
        self.ref_value_model = ui.SimpleStringModel(
            str(ref)
        )

        # Column 2
        self.tra_value_model = ui.SimpleStringModel(
            str(tra)
        )

        # Column 3
        self.display_color_model = ui.SimpleStringModel(
            str(display_color)
        )


# ============================================================
# Table Model
# ============================================================

class TableModel(ui.AbstractItemModel):
    """
    TreeView 数据模型。
    """

    def __init__(
        self,
        headers,
        data
    ):
        super().__init__()

        self.headers = headers

        self._items = [
            TableItem(*row)
            for row in data
        ]

    # --------------------------------------------------------
    # TreeView hierarchy
    # --------------------------------------------------------

    def get_item_children(
        self,
        item
    ):
        return self._items if item is None else []

    # --------------------------------------------------------
    # Column count
    # --------------------------------------------------------

    def get_item_value_model_count(
        self,
        item
    ):
        return len(self.headers)

    # --------------------------------------------------------
    # Column value model
    # --------------------------------------------------------

    def get_item_value_model(
        self,
        item,
        column_id
    ):

        if item is None:
            return None

        if column_id == 0:
            return item.name_model

        elif column_id == 1:
            return item.ref_value_model

        elif column_id == 2:
            return item.tra_value_model

        elif column_id == 3:
            return item.display_color_model

        return None

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset_data(
        self,
        new_data_list
    ):

        self._items.clear()

        for row in new_data_list:

            if len(row) >= 4:

                self._items.append(
                    TableItem(
                        row[0],
                        row[1],
                        row[2],
                        row[3]
                    )
                )

            elif len(row) == 3:

                self._items.append(
                    TableItem(
                        row[0],
                        row[1],
                        row[2],
                        "1,1,1"
                    )
                )

        self._item_changed(None)

    # --------------------------------------------------------
    # Add row
    # --------------------------------------------------------

    def add_row(
        self,
        name,
        ref,
        tra,
        display_color="1,1,1"
    ):

        self._items.append(
            TableItem(
                name,
                ref,
                tra,
                display_color
            )
        )

        self._item_changed(None)

    # --------------------------------------------------------
    # Remove
    # --------------------------------------------------------

    def remove_items(
        self,
        items_to_delete
    ):

        dirty = False

        for item in items_to_delete:

            if item in self._items:

                self._items.remove(item)

                dirty = True

        if dirty:

            self._item_changed(None)


# ============================================================
# Table Delegate
# ============================================================

class TableDelegate(ui.AbstractItemDelegate):

    def __init__(
        self,
        model
    ):

        super().__init__()

        self._model = model

        # ----------------------------------------------------
        # TreeView reference
        # ----------------------------------------------------

        self._tree_view = None

        # ----------------------------------------------------
        # 当前选中的 Item
        # ----------------------------------------------------

        self._selected_item = None

        # ----------------------------------------------------
        # 当前正在编辑的 Cell
        #
        # (item, column_id)
        # ----------------------------------------------------

        self._editing_item = None
        self._editing_column = None

        # ----------------------------------------------------
        # Widget references
        #
        # key:
        #     (id(item), column_id)
        #
        # value:
        #     dict
        #
        # ----------------------------------------------------

        self._cell_widgets = {}

        # ----------------------------------------------------
        # Value subscriptions
        # ----------------------------------------------------

        self._subscriptions = {}

        # ----------------------------------------------------
        # Row height
        # ----------------------------------------------------

        self.label_height = 30

    # ========================================================
    # Bind TreeView
    # ========================================================

    def bind_tree_view(
        self,
        tree_view
    ):
        """
        将 Delegate 与 TreeView 绑定。

        TreeView 负责真正的 Selection。
        Delegate 只响应 Selection 状态变化。
        """

        self._tree_view = tree_view

        # TreeView 原生 Selection callback
        tree_view.set_selection_changed_fn(
            self._on_selection_changed
        )

    # ========================================================
    # TreeView Selection Changed
    # ========================================================

    def _on_selection_changed(
        self,
        selections
    ):
        """
        TreeView 原生 Selection 回调。

        selections:
            List[AbstractItem]

        通常单选模式下只会包含一个 Item。
        """

        # ----------------------------------------------------
        # 没有选择
        # ----------------------------------------------------

        if not selections:

            self._selected_item = None

            self._editing_item = None
            self._editing_column = None

            self._update_all_cells()

            return

        # ----------------------------------------------------
        # 当前选中的 Item
        # ----------------------------------------------------

        selected_item = selections[0]

        # ----------------------------------------------------
        # 如果切换到了新的 Item
        # ----------------------------------------------------

        if selected_item is not self._selected_item:

            # 结束之前的编辑
            self._finish_edit()

            self._selected_item = selected_item

            # ------------------------------------------------
            # 注意：
            #
            # 这里只更新选中状态。
            #
            # 具体哪一列进入编辑，由 Cell click 决定。
            #
            # ------------------------------------------------

            self._update_all_cells()

    # ========================================================
    # Header
    # ========================================================

    def build_header(
        self,
        column_id
    ):

        title = ""

        if (
            self._model is not None
            and column_id < len(self._model.headers)
        ):
            title = self._model.headers[column_id]

        with ui.ZStack(
            height=self.label_height
        ):

            ui.Rectangle(
                style={
                    "background_color": COLOR_HEADER,
                    "border_color": COLOR_HEADER_BORDER,
                    "border_width": 1,
                }
            )

            ui.Label(
                title,
                alignment=ui.Alignment.CENTER,
                style={
                    "color": COLOR_HEADER_TEXT,
                    "font_size": 14,
                }
            )

    # ========================================================
    # Branch
    # ========================================================

    def build_branch(
        self,
        model,
        item,
        column_id,
        level,
        expanded
    ):
        pass

    # ========================================================
    # Build Cell
    # ========================================================

    def build_widget(
        self,
        model,
        item,
        column_id,
        level,
        expanded
    ):

        if item is None:
            return

        value_model = model.get_item_value_model(
            item,
            column_id
        )

        if value_model is None:
            return

        # ====================================================
        # Cell key
        # ====================================================

        cell_key = (
            id(item),
            column_id
        )

        # ====================================================
        # 当前是否正在编辑
        # ====================================================

        is_editing = (
            self._editing_item is item
            and self._editing_column == column_id
        )

        # ====================================================
        # Cell
        # ====================================================

        with ui.ZStack(
            height=self.label_height
        ):

            # ------------------------------------------------
            # Background
            # ------------------------------------------------

            ui.Rectangle(
                style={
                    "background_color": (
                        COLOR_EDITING
                        if is_editing
                        else COLOR_NORMAL
                    ),
                    "border_color": COLOR_BORDER,
                    "border_width": 1,
                }
            )

            # =================================================
            # Column 0 ~ 2
            # =================================================

            if column_id in (0, 1, 2):

                self._build_text_cell(
                    item,
                    column_id,
                    value_model,
                    is_editing,
                    cell_key
                )

            # =================================================
            # Column 3
            # =================================================

            elif column_id == 3:

                self._build_color_cell(
                    item,
                    value_model,
                    cell_key
                )

    # ========================================================
    # Text Cell
    # ========================================================

    def _build_text_cell(
        self,
        item,
        column_id,
        value_model,
        is_editing,
        cell_key
    ):

        # ========================================================
        # Column 0: Parameter Name
        #
        # 第一列永远不可编辑
        # ========================================================

        if column_id == 0:

            with ui.HStack(
                spacing=0
            ):

                ui.Spacer(
                    width=6
                )

                ui.Label(
                    value_model.as_string,
                    alignment=ui.Alignment.LEFT_CENTER,
                )

                ui.Spacer(
                    width=6
                )

            # ----------------------------------------------------
            # Name 发生变化时，只更新 Label
            # ----------------------------------------------------

            sub_key = f"label_{id(item)}_{column_id}"

            old_sub = self._subscriptions.pop(
                sub_key,
                None
            )

            if old_sub is not None:

                try:
                    old_sub.unsubscribe()
                except Exception:
                    pass

            subscription = (
                value_model.subscribe_value_changed_fn(
                    lambda model,
                    l=None:
                    None
                )
            )

            # Name 与csv文件名绑定，不能修改
            
            return

        # ========================================================
        # Column 1 / 2
        #
        # Reference / Transmission
        # ========================================================

        if is_editing:

            with ui.HStack(
                spacing=0
            ):

                ui.Spacer(
                    width=4
                )

                field = ui.StringField(
                    value_model,
                    height=self.label_height - 2,
                    style={
                        "background_color": COLOR_EDITING,
                        "border_width": 0,
                    }
                )

                ui.Spacer(
                    width=4
                )

            self._cell_widgets[cell_key] = {
                "field": field,
                "item": item,
                "column": column_id,
            }

            sub_key = (
                f"edit_{id(item)}_{column_id}"
            )

            old_sub = self._subscriptions.pop(
                sub_key,
                None
            )

            if old_sub is not None:

                try:
                    old_sub.unsubscribe()
                except Exception:
                    pass

            subscription = (
                value_model.subscribe_end_edit_fn(
                    lambda model,
                    i=item,
                    c=column_id:
                    self._on_field_end_edit(
                        i,
                        c,
                        model
                    )
                )
            )

            self._subscriptions[sub_key] = subscription

            self._focus_field_next_frame(
                field
            )

        # ========================================================
        # Normal state
        # ========================================================

        else:

            with ui.HStack(
                spacing=0
            ):

                ui.Spacer(
                    width=6
                )

                label = ui.Label(
                    value_model.as_string,
                    alignment=ui.Alignment.LEFT_CENTER,
                )

                ui.Spacer(
                    width=6
                )

            sub_key = (
                f"label_{id(item)}_{column_id}"
            )

            old_sub = self._subscriptions.pop(
                sub_key,
                None
            )

            if old_sub is not None:

                try:
                    old_sub.unsubscribe()
                except Exception:
                    pass

            subscription = (
                value_model.subscribe_value_changed_fn(
                    lambda model,
                    l=label:
                    self._update_label(
                        model,
                        l
                    )
                )
            )

            self._subscriptions[sub_key] = subscription

            # ----------------------------------------------------
            # 只有 Reference / Transmission 可以点击编辑
            # ----------------------------------------------------

            label.set_mouse_pressed_fn(
                lambda x,
                y,
                button,
                modifier,
                i=item,
                c=column_id:
                self._on_cell_pressed(
                    i,
                    c,
                    button
                )
            )

    # ========================================================
    # Color Cell
    # ========================================================

    def _build_color_cell(
        self,
        item,
        value_model,
        cell_key
    ):

        with ui.HStack(
            spacing=6
        ):

            ui.Spacer(
                width=8
            )

            # ------------------------------------------------
            # Color preview
            # ------------------------------------------------

            color_preview = ui.Rectangle(
                width=18,
                height=18
            )

            self._update_color_preview(
                value_model,
                color_preview
            )

            # ------------------------------------------------
            # RGB text
            # ------------------------------------------------

            label = ui.Label(
                value_model.as_string,
                alignment=ui.Alignment.LEFT_CENTER
            )

            ui.Spacer(
                width=4
            )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        self._cell_widgets[cell_key] = {
            "color_preview": color_preview,
            "label": label,
            "item": item,
            "column": 3,
        }

        # ----------------------------------------------------
        # RGB Model changed
        # ----------------------------------------------------

        sub_key = f"color_{id(item)}"

        old_sub = self._subscriptions.pop(
            sub_key,
            None
        )

        if old_sub is not None:

            try:
                old_sub.unsubscribe()
            except Exception:
                pass

        subscription = (
            value_model.subscribe_value_changed_fn(
                lambda model,
                rect=color_preview,
                l=label:
                self._on_color_changed(
                    model,
                    rect,
                    l
                )
            )
        )

        self._subscriptions[sub_key] = subscription

        # ----------------------------------------------------
        # 点击 Color
        # ----------------------------------------------------

        label.set_mouse_pressed_fn(
            lambda x,
            y,
            button,
            modifier,
            i=item:
            self._on_color_pressed(
                i,
                button
            )
        )

        color_preview.set_mouse_pressed_fn(
            lambda x,
            y,
            button,
            modifier,
            i=item:
            self._on_color_pressed(
                i,
                button
            )
        )

    # ========================================================
    # Cell pressed
    # ========================================================

    def _on_cell_pressed(
        self,
        item,
        column_id,
        button
    ):

        if button != 0:
            return

        # ----------------------------------------------------
        # 先让 TreeView 负责 Selection
        # ----------------------------------------------------

        if self._tree_view is not None:

            # 已经是当前选择
            if self._selected_item is not item:

                self._tree_view.selection = [item]

        # ----------------------------------------------------
        # 记录要编辑的列
        # ----------------------------------------------------

        self._editing_item = item
        self._editing_column = column_id

        # ----------------------------------------------------
        # TreeView Selection 改变后会 dirty widget。
        #
        # 如果当前 item 本来就是 selected，
        # Selection callback 不一定会再次触发。
        #
        # 因此这里主动刷新。
        # ----------------------------------------------------

        self._update_all_cells()

    # ========================================================
    # Color pressed
    # ========================================================

    def _on_color_pressed(
        self,
        item,
        button
    ):

        if button != 0:
            return

        # ----------------------------------------------------
        # Selection
        # ----------------------------------------------------

        if self._tree_view is not None:

            if self._selected_item is not item:

                self._tree_view.selection = [item]

        # ----------------------------------------------------
        # 当前编辑列
        # ----------------------------------------------------

        self._editing_item = item
        self._editing_column = 3

        # ----------------------------------------------------
        # 打开颜色选择器
        # ----------------------------------------------------

        value_model = (
            item.display_color_model
        )

        try:

            RGBColorPickerDialog(
                value_model
            )

        except Exception as e:

            print(
                "[TableDelegate] "
                f"Failed to open RGBColorPickerDialog: {e}"
            )

        # ----------------------------------------------------
        # 更新 UI
        # ----------------------------------------------------

        self._update_all_cells()

    # ========================================================
    # Selection / Editing refresh
    # ========================================================

    def _update_all_cells(self):

        """
        让 TreeView 在下一帧重新生成 Delegate widgets。
        """

        if self._tree_view is not None:

            self._tree_view.dirty_widgets()

    # ========================================================
    # End edit
    # ========================================================

    def _on_field_end_edit(
        self,
        item,
        column_id,
        model
    ):

        # ----------------------------------------------------
        # 结束编辑
        # ----------------------------------------------------

        if (
            self._editing_item is item
            and self._editing_column == column_id
        ):

            self._editing_item = None
            self._editing_column = None

        # ----------------------------------------------------
        # 刷新
        # ----------------------------------------------------

        self._update_all_cells()

    # ========================================================
    # Finish current edit
    # ========================================================

    def _finish_edit(self):

        if self._editing_item is None:
            return

        item = self._editing_item
        column = self._editing_column

        # ----------------------------------------------------
        # 找到当前 field
        # ----------------------------------------------------

        cell_key = (
            id(item),
            column
        )

        cell_info = self._cell_widgets.get(
            cell_key
        )

        if cell_info is not None:

            field = cell_info.get(
                "field"
            )

            if field is not None:

                try:

                    # 让 Field 正常结束编辑
                    field.model.end_edit()

                except Exception:
                    pass

        # ----------------------------------------------------
        # 清理状态
        # ----------------------------------------------------

        self._editing_item = None
        self._editing_column = None

    # ========================================================
    # Update label
    # ========================================================

    @staticmethod
    def _update_label(
        model,
        label
    ):

        label.text = model.as_string

    # ========================================================
    # Update color
    # ========================================================

    def _update_color_preview(
        self,
        model,
        rect
    ):

        try:

            parts = [
                float(x.strip())
                for x in model.as_string.split(",")
            ]

            if len(parts) >= 3:

                r = max(
                    0.0,
                    min(1.0, parts[0])
                )

                g = max(
                    0.0,
                    min(1.0, parts[1])
                )

                b = max(
                    0.0,
                    min(1.0, parts[2])
                )

                rect.style = {
                    "background_color": ui.color(
                        r,
                        g,
                        b,
                        1.0
                    ),
                    "border_radius": 3,
                }

        except Exception as e:

            print(
                "[TableDelegate] "
                f"Color parse error: {e}"
            )

            rect.style = {
                "background_color": 0xFF888888
            }

    # ========================================================
    # Color changed
    # ========================================================

    def _on_color_changed(
        self,
        model,
        rect,
        label
    ):

        self._update_color_preview(
            model,
            rect
        )

        label.text = model.as_string

    # ========================================================
    # Focus Field
    # ========================================================

    def _focus_field_next_frame(
        self,
        field
    ):

        import asyncio
        import omni.kit.app

        async def focus():

            try:

                await (
                    omni.kit.app
                    .get_app()
                    .next_update_async()
                )

                if field.visible:

                    field.focus_keyboard()

            except Exception as e:

                print(
                    "[TableDelegate] "
                    f"Focus error: {e}"
                )

        asyncio.ensure_future(
            focus()
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def destroy(self):

        # ----------------------------------------------------
        # Unsubscribe
        # ----------------------------------------------------

        for subscription in (
            self._subscriptions.values()
        ):

            try:
                subscription.unsubscribe()
            except Exception:
                pass

        self._subscriptions.clear()

        self._cell_widgets.clear()

        self._tree_view = None
        self._selected_item = None
        self._editing_item = None
        self._editing_column = None


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