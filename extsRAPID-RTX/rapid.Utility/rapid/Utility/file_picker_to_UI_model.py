import os
import omni.usd
import omni.kit.commands
import carb
from pxr import Sdf
from omni.kit.window.filepicker import FilePickerDialog


class FilePickerHelper:
    def __init__(self, model_to_update=None, default_path: str = None, import_to_stage: bool = False,
                 item_filter=[(".usd, .usda, .usdc", "USD Files")]):
        """
        Args:
            model_to_update: ui.SimpleStringModel, 路径更新的目标模型
            default_path: str, 默认打开的文件夹路径
            import_to_stage: bool, 如果识别到是文件，是否自动引入舞台
            item_filter: List[str]: 文件过滤
        """
        self._model = model_to_update
        self._default_path = default_path.replace("\\", "/") if default_path else None
        self._import_to_stage = import_to_stage
        self.item_filter = item_filter
        self._file_picker = None

    def select_file_or_folder(self):
        """打开文件选择器"""
        # 保留 USD 过滤器
        # 因为在 FilePicker 中，文件夹总是可见的
        self._file_picker = FilePickerDialog(
            "Select File or Folder",
            allow_multi_selection=False,
            apply_button_label="Select",
            click_apply_handler=self._on_item_selected,
            item_filter_options=self.item_filter,
            current_directory=self._default_path
        )
        self._file_picker.show()

    def _on_item_selected(self, filename: str, dirname: str):
        """核心逻辑：自动识别并提取路径"""
        if not dirname:
            self._close_picker()
            return

        # 1. 尝试拼接完整路径
        dirname = dirname.replace("\\", "/")
        if filename:
            full_path = os.path.join(dirname, filename).replace("\\", "/")
        else:
            # 如果 filename 为空，说明用户直接选择了当前浏览的文件夹
            full_path = dirname

        # 2. 判断是文件还是文件夹
        if os.path.isdir(full_path):
            # --- 情况 A: 识别为文件夹 ---
            print(f"[AssetImportHelper] Folder selected: {full_path}")
            # 文件夹不执行 import_to_stage 逻辑
        elif os.path.isfile(full_path):
            # --- 情况 B: 识别为文件 ---
            print(f"[AssetImportHelper] File selected: {full_path}")
            if self._import_to_stage:
                # 只有文件才尝试引入舞台
                self.import_usd_as_instance(full_path)
        else:
            # --- 情况 C: 可能是 Nucleus 路径或其他特殊路径 ---
            # 对于 omniverse:// 协议，os.path 无法判断，默认作为路径返回
            print(f"[AssetImportHelper] Protocol path selected: {full_path}")

        # 3. 更新 UI Model
        if self._model:
            self._model.set_value(full_path)

        self._close_picker()

    def _close_picker(self):
        if self._file_picker:
            self._file_picker.hide()
            self._file_picker = None

    @staticmethod
    def import_usd_as_instance(usd_path: str) -> str:
        """将本地 USD 导入为 Instanceable (内部包含安全检查)"""
        # 再次确保不是文件夹
        if os.path.isdir(usd_path):
            return ""

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return ""

        base_name = os.path.splitext(os.path.basename(usd_path))[0].replace(".", "_")
        prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/{base_name}", False)

        success = omni.kit.commands.execute(
            "CreateReferenceCommand",
            usd_context=omni.usd.get_context(),
            path_to=Sdf.Path(prim_path),
            asset_path=usd_path,
            instanceable=True
        )
        return prim_path if success else ""