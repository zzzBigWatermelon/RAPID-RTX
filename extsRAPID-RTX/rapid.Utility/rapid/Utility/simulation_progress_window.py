import omni.ui as ui
import omni.kit.notification_manager as nm
import platform
import os
from .project_validity_check import get_current_usd_path, quick_project_check, get_folder


class SimulationProgressWindow(ui.Window):
    def __init__(self, total_steps=100, **kwargs):
        # 创建一个不可调整大小、位于中心的窗口
        super().__init__("Simulation Progress", width=500, height=250,
                         dockPreference=ui.DockPreference.DISABLED,
                         flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)

        self.total_steps = total_steps

        with self.frame:
            with ui.VStack(padding=20, spacing=15):
                self.status_label = ui.Label("Initializing simulation...", word_wrap=True, style={"color": 0xFFAAAAAA, "font_size": 24})

                # 进度条
                self.progress_bar = ui.ProgressBar()
                self.progress_model = self.progress_bar.model
                self.progress_model.set_value(0.0)

                self.detail_label = ui.Label("Step 0 of 0", style={"color": 0xFFAAAAAA, "font_size": 24})
                self.elapsed_label = ui.Label("", style={"color": 0xFFAAAAAA, "font_size": 24})

                with ui.HStack(height=40):
                    ui.Spacer()                     # 左侧弹性空间
                    self.open_folder_btn = ui.Button("Open Result Folder", width=130, clicked_fn=self._on_open_folder, style={"font_size": 20, "padding": "8px 12px"})
                    ui.Spacer(width=30)             # 两个按钮之间的固定间距
                    self.close_btn = ui.Button("Cancel", width=130, clicked_fn=self._on_cancel, style={"font_size": 20, "padding": "8px 12px"})
                    ui.Spacer()                     # 右侧弹性空间

    def update_progress(self, current_step, message=None):
        """更新进度条和文字"""
        progress = current_step / self.total_steps
        self.progress_model.set_value(progress)
        self.detail_label.text = f"Step {current_step} of {self.total_steps}"
        if message:
            self.status_label.text = message

        # 如果完成了
        if current_step >= self.total_steps:
            self.status_label.text = "Simulation Completed Successfully!"
            self.close_btn.text = "Close"

    def set_elapsed_time(self, elapsed_seconds):
        self.elapsed_label.text = f"Total time: {elapsed_seconds:.2f} seconds"

    def _on_open_folder(self):
        # 这里首先检查项目文件的完整性
        current_usd_parent_dir = get_current_usd_path()
        if not current_usd_parent_dir:
            nm.post_notification("Please open a valid project..", status=nm.NotificationStatus.WARNING, duration=5)
        if not quick_project_check():
            return None

        # 获取参数文件夹
        result_dir = get_folder("result")
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

    def _on_cancel(self):
        # 实际开发中可以这里触发仿真的停止逻辑
        self.visible = False
