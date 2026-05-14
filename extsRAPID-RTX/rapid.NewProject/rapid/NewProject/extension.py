import os
from pathlib import Path
import omni.ext
import omni.ui as ui
import shutil
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items, add_layout, remove_layout
from omni.kit.menu.utils.layout import MenuLayout
from omni.kit.window.filepicker import FilePickerDialog
from rapid.Utility.project_validity_check import SUPPORTED_FOLDERS

# 默认舞台usd文件
DEFAULT_STAGE = str(Path(__file__).parent.parent.parent/'data'/'default_stage.usd')
DEFAULT_PARAMETERS = str(Path(__file__).parent.parent.parent/'data'/'simulation_parameters.json')


class NewProjectExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[rapid.NewProject] rapid NewProject startup")
        self._menu_items = []
        self._file_picker = None  # 保持对 picker 的引用防止被垃圾回收

        # --- 1. 定义菜单项 ---
        self._menu_list = [
            MenuItemDescription(name="New Project", glyph="folder.svg", onclick_fn=self._on_menu_open_picker)
        ]
        # 添加到顶部菜单栏
        add_menu_items(self._menu_list, "File")

        # 我们创建一个布局规则，告诉系统 "New Project" 应该排在哪里
        self._layout = [
            MenuLayout.Menu("File", [
                MenuLayout.Item("New Project"),  # 排在第1个
                # 确保New Project在 "New" 和 "Open" 前面，显式把它们写在后面：
                MenuLayout.Item("New"),         # 排在第2个 (官方菜单项)
                MenuLayout.Item("Open"),        # 排在第3个 (官方菜单项)
            ])
        ]

        # 4. 应用排序规则
        add_layout(self._layout)

    def on_shutdown(self):
        # 移除排序
        if self._layout:
            remove_layout(self._layout)
            self._layout = None
        # 清理菜单
        remove_menu_items(self._menu_list, "File")
        # 清理 picker
        if self._file_picker:
            self._file_picker.hide()
            self._file_picker = None

        print("[rapid.NewProject] rapid NewProject shutdown")

    def _on_menu_open_picker(self):
        '''点击菜单触发：打开对话框'''
        # 创建文件选择对话框
        self._file_picker = FilePickerDialog(
            "Select Directory",  # 弹窗标题
            allow_multi_selection=False,
            apply_button_label="Select",
            click_apply_handler=self._on_folder_selected,  # 选中后的回调
            item_filter_options=None,  # 不过滤，显示所有
        )
        # 注意：Omniverse 的 FilePicker 比较灵活，通常用来选文件，
        # 如果要选文件夹，用户通常是进入该文件夹后点击 Select。
        self._file_picker.show()

    def _on_folder_selected(self, filename: str, dirname: str):
        """
        选中后的回调函数
        参数:
            filename (自动传参): 用户选中的文件名（如果是选文件夹，这通常是空的或者是文件夹名）
            dirname (自动传参): 目录路径
        """
        if filename:
            full_path = os.path.join(dirname, filename)
        else:
            full_path = dirname

        # 路径标准化（防止 Windows 反斜杠问题）
        full_path = full_path.replace("\\", "/")
        print(f"用户选中的路径: {full_path}")

        self._file_picker.hide()
        self._file_picker = None

        # 执行后续逻辑，创建目录结构并生成/打开 USD 文件
        self.create_project_structure(full_path, filename)

    def create_project_structure(self, full_path, filename):
        """
        核心功能：创建目录结构并生成/打开 USD 文件
        """
        print(f"[Rapid] 开始创建项目结构: {full_path}")

        try:
            # 如果不存在，创建主文件夹
            if not os.path.exists(full_path):
                os.makedirs(full_path, exist_ok=True)

            # 创建三个子文件夹
            sub_folders = [SUPPORTED_FOLDERS["intermediate_data"],
                           SUPPORTED_FOLDERS["parameters"],
                           SUPPORTED_FOLDERS["result"]]
            for sub in sub_folders:
                sub_path = os.path.join(full_path, sub)
                os.makedirs(sub_path, exist_ok=True)

            # -------------创建默认舞台和保存默认舞台文件-----------------
            # 确定 USD 文件名
            # 如果 filename_input 为空，使用 "New Project"
            # 如果 full_path 本身就是项目名（例如 path/to/MyBot），我们也可以取 basename
            if filename:
                base_name = filename
            else:
                # 尝试从路径获取文件夹名，如果路径是根目录等特殊情况，回退到 New Project
                base_name = os.path.basename(full_path.rstrip("/"))
                if not base_name:
                    base_name = "New Project"

            # 确保后缀名为 .usd（防止filename自带.usd后缀）
            if not base_name.lower().endswith(".usd"):
                usd_filename = f"{base_name}.usd"
            else:
                usd_filename = base_name

            usd_full_path = os.path.join(full_path, usd_filename)
            usd_full_path = usd_full_path.replace("\\", "/")  # USD API 偏好正斜杠

            # 打开默认的USD舞台文件
            ctx = omni.usd.get_context()
            ctx.open_stage(DEFAULT_STAGE)

            # 这一步相当于 "创建新文件" + "打开文件"
            success = ctx.save_as_stage(usd_full_path)

            if success:
                print(f"[Rapid] 成功创建并打开 USD: {usd_full_path}")
                # 可选：发送通知给 UI
                # omni.kit.window.status_bar.post_message(f"Project Created: {usd_filename}")
            else:
                print(f"[Rapid] [Error] USD 保存失败: {usd_full_path}")

            # ---------------------创建模拟参数文件---------------------
            parameters_path = os.path.join(full_path, SUPPORTED_FOLDERS["parameters"])
            shutil.copy2(DEFAULT_PARAMETERS, parameters_path)

        except Exception as e:
            print(f"[Rapid] [Error] 创建项目失败: {e}")
            import traceback
            traceback.print_exc()
