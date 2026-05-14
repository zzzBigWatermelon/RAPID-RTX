import omni.ui as ui


class DefineBandsWindow:
    def __init__(self, ui_model):
        # 模拟主窗口中已有的 models
        self.models = {
            "bands": ui.SimpleStringModel("400, 500, 600")
        }
        self.label_width = 150
        self._define_bands_window = None  # 用于存储新窗口实例

    def manage_bands_fn(self):
        '''对波段信息的操作'''
        LABEL_HEIGHT = 22
        with ui.HStack(height=LABEL_HEIGHT):
            ui.Label("Spectral Bands:", width=self.label_width)
            ui.StringField(model=self.models["bands"])

            ui.Spacer(width=10)
            ui.Button("Define Bands", width=100, clicked_fn=self.on_define_bands)

    def on_define_bands(self):
        """点击按钮弹出新窗口"""
        # 如果窗口已存在，直接显示并置顶
        if self._define_bands_window:
            self._define_bands_window.visible = True
            self._define_bands_window.focus()
            return

        # 创建新窗口
        self._define_bands_window = ui.Window("Define New Bands", width=400, height=450)

        # 定义临时存储数据的 models
        center_models = [ui.SimpleFloatModel(0.0) for _ in range(4)]
        overwrite_model = ui.SimpleBoolModel(True)
        append_model = ui.SimpleBoolModel(False)

        # 互斥逻辑：勾选一个，取消另一个
        def on_overwrite_click(m):
            if m.as_bool:
                append_model.set_value(False)

        def on_append_click(m):
            if m.as_bool:
                overwrite_model.set_value(False)
        overwrite_model.add_value_changed_fn(on_overwrite_click)
        append_model.add_value_changed_fn(on_append_click)

        with self._define_bands_window.frame:
            with ui.VStack(spacing=10, padding=15):
                # --- 数据输入区 ---
                for i in range(4):
                    with ui.VStack(spacing=2):
                        ui.Label(f"Observation Center {i+1}:", style={"color": 0xFFFFFF00}) # 黄色标题
                        with ui.HStack():
                            ui.Label("X [m]:", width=self.label_width)
                            ui.FloatField(model=center_models[i])

                ui.Spacer(height=10)
                ui.Line(style={"color": 0x33FFFFFF})  # 分割线

                # --- 勾选框区 ---
                with ui.HStack(height=20):
                    ui.CheckBox(model=overwrite_model)
                    ui.Label("Overwrite existing bands")
                    ui.Spacer(width=20)
                    ui.CheckBox(model=append_model)
                    ui.Label("Append to existing bands")
                ui.Spacer(height=20)

                # --- 按钮区 ---
                with ui.HStack(spacing=15, height=30):
                    def on_confirm():
                        # 获取新输入的数值
                        new_vals = [str(round(m.as_float, 3)) for m in center_models]
                        new_str = ", ".join(new_vals)

                        if overwrite_model.as_bool:
                            # 覆盖
                            self.models["bands"].set_value(new_str)
                        else:
                            # 追加
                            current = self.models["bands"].as_string
                            if current:
                                self.models["bands"].set_value(f"{current}, {new_str}")
                            else:
                                self.models["bands"].set_value(new_str)

                        self._define_bands_window.visible = False

                    def on_cancel():
                        self._define_bands_window.visible = False

                    ui.Button("Confirm", clicked_fn=on_confirm, style={"background_color": 0xFF448844})
                    ui.Button("Cancel", clicked_fn=on_cancel)

        # 当窗口被用户点叉关闭时的回调
        self._define_bands_window.set_visibility_changed_fn(lambda v: self._on_window_close(v))

    def _on_window_close(self, visible):
        if not visible:
            self._define_bands_window.visible = False

# --- 测试运行 (在 Script Editor 中使用时) ---
ext = DefineBandsWindow()
main_win = ui.Window("Main", width=500, height=200)
with main_win.frame:
    with ui.VStack():
        ext.manage_bands_fn()
