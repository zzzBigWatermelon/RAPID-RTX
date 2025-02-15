'''此模块将用于生成BRF曲线的图片展示窗口'''
import omni.kit.ui
import omni.ui as ui
from pathlib import Path
import os
WINDOW_TITLE = "BRF"


# 这个代码原本应该是将一个窗口放到Window菜单下，可以开启或隐藏
# 现在用它来直接建立一个窗口来展示.extension.plot()方法生成的曲线图
class BRF_plotExtension(ui.Window):
    def __init__(self, title) -> None:
        super().__init__(title, width=600, height=500)
        self.set_visibility_changed_fn(self._on_visibility_changed)
        self._build_ui()

    def on_shutdown(self):
        self._win = None

    def show(self):
        self.visible = True
        self.focus()

    def hide(self):
        self.visible = False

    def _build_ui(self):
        imageGray = Path(__file__).parent.parent/'result'/'BRF'/'BRF.png'
        absPathGray = os.path.abspath(imageGray)
        imageReflectance = Path(__file__).parent.parent/'result'/'BRF'/'BRF1.png'
        absPathReflectance = os.path.abspath(imageReflectance)
        with self.frame:
            ui.Image(
                absPathGray,
                fil_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                alignment=ui.Alignment.CENTER
            )
            ui.Image(
                absPathReflectance,
                fil_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                alignment=ui.Alignment.CENTER
            )

    def _on_visibility_changed(self):
        omni.kit.ui.get_editor_menu().set_value()
