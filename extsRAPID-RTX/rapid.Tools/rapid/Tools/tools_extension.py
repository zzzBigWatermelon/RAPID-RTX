import omni.ext
import omni.ui as ui
from omni.kit.menu.utils import MenuItemDescription, add_menu_items
import os
import platform
import omni.kit.notification_manager as nm
from rapid.Utility import project_validity_check  # 项目有效性检查与文件路径获取
from pathlib import Path
from .image_viewer import MyImageViewerExtension
from .pointcloud_viewer import PointCloudViewerExtension


class ToolsExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        print("[rapid.Tools] rapid Tools startup")
        # 创建数据查看器实例
        self._image_viewer = MyImageViewerExtension()
        self._pointcloud_viewer = PointCloudViewerExtension()

        # 定义一级菜单的名称
        self.TOP_MENU_NAME = "Tool"
        # 定义菜单项
        self._menu_list = [
            MenuItemDescription(name="Open Results Folder", glyph="folder.svg", onclick_fn=self._on_open_result_button_clicked),
            MenuItemDescription(name="Open Custom Script Folder", glyph="folder.svg", onclick_fn=self.on_open_script_folder_button_clicked),
            MenuItemDescription(name="Image Data Viewer", glyph="folder.svg", onclick_fn=self._on_image_viewer_button_clicked),
            MenuItemDescription(name="Pointcloud Data Viewer", glyph="folder.svg", onclick_fn=self._on_pointcloud_data_button_clicked),
            MenuItemDescription(name="Batch Tool", glyph="folder.svg", onclick_fn=self._on_save_button_clicked),]
        # 添加到顶部菜单栏
        add_menu_items(self._menu_list, self.TOP_MENU_NAME)

    def _on_open_result_button_clicked(self):
        # 这里首先检查项目文件的完整性
        current_usd_parent_dir = project_validity_check.get_current_usd_path()
        if not current_usd_parent_dir:
            nm.post_notification("Please open a valid project..", status=nm.NotificationStatus.WARNING, duration=5)
        if not project_validity_check.quick_project_check():
            return None

        # 获取参数文件夹
        result_dir = project_validity_check.get_folder("result")
        # 3. 根据操作系统打开文件夹
        # Windows 系统使用 os.startfile
        if platform.system() == "Windows":
            # 将路径转换为 Windows 格式（反斜杠）
            norm_path = os.path.normpath(result_dir)
            os.startfile(norm_path)
            print(f"[rapid.Tools] Opening folder: {norm_path}")

        # 如果你以后要在 Linux 上运行 Isaac Sim (Ubuntu)
        else:
            import subprocess
            subprocess.Popen(['xdg-open', result_dir])

    def on_open_script_folder_button_clicked(self):
        custom_script_folder_path = Path(__file__).parent.parent.parent/'data'/'custom_script'
        # Windows 系统使用 os.startfile
        if platform.system() == "Windows":
            # 将路径转换为 Windows 格式（反斜杠）
            norm_path = os.path.normpath(custom_script_folder_path)
            os.startfile(norm_path)
            print(f"[rapid.Tools] Opening folder: {norm_path}")

        # 如果你以后要在 Linux 上运行 Isaac Sim (Ubuntu)
        else:
            import subprocess
            subprocess.Popen(['xdg-open', custom_script_folder_path])

    def _on_image_viewer_button_clicked(self):
        self._image_viewer.show_window()

    def _on_pointcloud_data_button_clicked(self):
        self._pointcloud_viewer.show_window()

    def _on_save_button_clicked(self):
        pass

    def on_shutdown(self):
        omni.kit.menu.utils.remove_menu_items(self._menu_list, self.TOP_MENU_NAME)
        print("[rapid.Tools] rapid Tools shutdown")
