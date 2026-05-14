import os
import omni.usd
import omni.kit.notification_manager as nm
from pathlib import Path
import re

# 项目下的次级文件夹

SUPPORTED_FOLDERS = {
    "intermediate_data": "intermediate_data",
    "parameters": "parameters",
    "result": "result",
    "cache": "cache",
    "logs": "logs",
    'data_assimilation': 'data_assimilation'
}


class ProjectContext:
    """项目管理上下文"""

    def __init__(self):
        self._current_usd_parent_dir = None

    def get_current_usd_path(self):
        """
        获取当前USD文件的父目录路径
        如果不是Windows本地路径(C:/...)则触发警告并返回None
        """
        stage_url = omni.usd.get_context().get_stage_url()

        # 检查是否为Windows本地路径
        # 匹配: C:/..., C:\..., file:///C:/..., file:///C:\...
        is_windows_path = bool(
            re.match(r'^[A-Za-z]:[/\\]', stage_url) or 
            re.match(r'^file:///[A-Za-z]:[/\\]', stage_url)
        )

        if not is_windows_path:
            self._current_usd_parent_dir = None
            nm.post_notification(
                "Please create a new project first.",
                "Please create a new project first.",
                status=nm.NotificationStatus.WARNING,
                duration=5
            )
            return None

        self._current_usd_parent_dir = str(Path(stage_url).parent).replace("\\", "/")
        return self._current_usd_parent_dir

    def get_folder(self, folder_name):
        """
        获取指定文件夹的完整路径
        参数:
            folder_name: 文件夹名称
        返回:
            完整路径或None
        """
        stage_url = omni.usd.get_context().get_stage_url()
        _current_usd_parent_dir = str(Path(stage_url).parent).replace("\\", "/")
        if not _current_usd_parent_dir:
            return None

        if folder_name not in SUPPORTED_FOLDERS:
            print(f"[Warning] Unsupported folder names: {folder_name}")
            return None

        folder_path = os.path.join(_current_usd_parent_dir,
                                   SUPPORTED_FOLDERS[folder_name])
        folder_path = folder_path.replace("\\", "/")

        return folder_path

    def quick_project_check(self):
        """
        快速检查必要的三个子文件夹
        如果验证失败，触发警告
        """
        # 获取并验证父目录
        current_dir = self.get_current_usd_path()
        if not current_dir:
            return None, None

        # 验证三个关键文件夹是否存在
        expected_subs = ["intermediate_data", "parameters", "result"]
        for sub in expected_subs:
            full_path = os.path.join(current_dir, sub)
            if not os.path.exists(full_path):
                nm.post_notification(
                    f"The project is missing necessary folders: {sub}, please check the open folder",
                    f"The project is missing necessary folders: {sub}, please check the open folder",
                    status=nm.NotificationStatus.WARNING,
                    duration=5
                )
                return False
        return True


# 单例实例
_project_context = ProjectContext()


def get_current_usd_path():
    """获取当前USD文件的父目录路径"""
    return _project_context.get_current_usd_path()


def get_folder(name):
    """返回指定的子目录"""
    return _project_context.get_folder(name)


def quick_project_check():
    """快速检查项目有效性并返回parameters和result路径"""
    return _project_context.quick_project_check()


# 简洁的测试
if __name__ == "__main__":
    # 测试获取父目录
    parent_dir = get_current_usd_path()
    if parent_dir:
        print(f"USD父目录: {parent_dir}")

        # 测试项目检查
        quick_project_check()
    intermediate_path = get_folder("intermediate_data")
    print(intermediate_path)
