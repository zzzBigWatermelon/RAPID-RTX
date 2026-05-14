import omni.ext
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
from pathlib import Path

# 导入拆分出去的逻辑函数
from .support_logic import on_about_button_clicked, on_manual_button_clicked, on_github_button_clicked


class HelpExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        print("[rapid.Help] rapid Help startup")

        # 初始化窗口对象为 None
        self._window = None

        # --- 定义资源路径 ---
        # 获取当前扩展的 data 目录绝对路径
        ext_path = Path(__file__).parent.parent.parent
        self._icon_path = str(ext_path / 'data' / 'RAPID-RTX.png')
        self._manual_path = ext_path / 'data' / 'RAPID-RTX_User_Manual.pdf'
        self._github_url = "https://github.com/zzzBigWatermelon"

        # --- 定义菜单项 ---
        # 使用 lambda 将 self (当前扩展实例) 传递给拆分出去的函数
        self._menu_list = [
            MenuItemDescription(
                name="About",
                onclick_fn=lambda: on_about_button_clicked(self)
            ),
            MenuItemDescription(
                name="User Manual", 
                onclick_fn=lambda: on_manual_button_clicked(self)
            ),
            MenuItemDescription(
                name="GitHub", 
                onclick_fn=lambda: on_github_button_clicked(self)
            ),
        ]

        # 添加到顶部菜单栏
        add_menu_items(self._menu_list, "Support")

    def on_shutdown(self):
        print("[rapid.Help] rapid Help shutdown")
        # 清理菜单
        remove_menu_items(self._menu_list, "Support")
        # 清理窗口
        if self._window:
            self._window.destroy()
            self._window = None