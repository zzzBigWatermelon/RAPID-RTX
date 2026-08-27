# support_logic.py
import omni.ui as ui
import webbrowser
from pathlib import Path


def on_about_button_clicked(ext):
    """
    ext: 传入的 HelpExtension 实例
    """
    # 如果窗口已存在，则将其提到最前
    if ext._window:
        ext._window.visible = True
        ext._window.focus()
        return

    # 创建窗口
    ext._window = ui.Window("About RAPID-RTX", width=500, height=450, 
                             flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_DOCKING)

    with ext._window.frame:
        with ui.VStack(spacing=5, margin=20):
            # --- 上部区域：图标 + 标题/版本 ---
            with ui.HStack(spacing=10):
                ui.Image(ext._icon_path, width=100, height=100, style={"border_radius": 10})

                with ui.VStack(spacing=5):
                    ui.Label("RAPID-RTX", style={"font_size": 28, "color": 0xFF00FF00, "font_weight": "bold"})
                    ui.Label("Version: 1.0.0", style={"color": 0xAAFFFFFF})
                    ui.Label("Date: 2026-08-27", style={"color": 0x88FFFFFF})

            ui.Line(style={"color": 0x33FFFFFF})

            # --- 中间区域：简介 ---
            ui.Label("Description:", style={"font_weight": "bold"})
            with ui.ScrollingFrame(height=120,
                                   style={"background_color": 0x22000000, 
                                          "border_color": 0x33FFFFFF, 
                                          "border_width": 1,
                                          "border_radius": 5}):
                with ui.VStack(margin=10):
                    description_text = (
                        'RAPID-RTX is a high-performance 3D Radiative Transfer Model (RTM) simulation platform based on NVIDIA Isaac Sim. '
                        'Developed in Python, it supports the rapid generation of high-fidelity remote sensing imagery and point cloud data in large and complex 3D environments. '
                        'The platform integrates a GPU-accelerated ray tracing engine, possessing multi-sensor, multi-angle, and multispectral simulation capabilities, '
                        'covering mainstream remote sensing payloads such as optical and lidar (point cloud/full waveform). It also includes a built-in AI-driven scene generation module, '
                        'significantly improving simulation efficiency and scene diversity.'
                    )
                    ui.Label(description_text, word_wrap=True, style={"color": 0xCCFFFFFF})

            ui.Spacer(height=10)

            # --- 下部区域：作者、邮箱、单位 ---
            with ui.VStack(spacing=4):
                with ui.HStack(height=0):
                    ui.Label("Authors:", width=80, style={"font_weight": "bold"})
                    ui.Label("zhuangzhuang zhang, huaguo huang*")

                with ui.HStack(height=0):
                    ui.Label("Email:", width=80, style={"font_weight": "bold"})
                    ui.Label("zzz_zhang666@163.com; huaguo_huang@bjfu.edu.cn")

                with ui.HStack(height=0):
                    ui.Label("Affiliation:", width=80, style={"font_weight": "bold"})
                    ui.Label("Beijing Forestry University")

                with ui.HStack(height=20): 
                    ui.Label("Website:", width=80, style={"font_weight": "bold"})
                    ui.Button(
                        "http://www.3dforest.cn/", 
                        clicked_fn=lambda: webbrowser.open("http://www.3dforest.cn/"),
                        width=0,
                        style={
                            "background_color": 0x0, "color": 0xFF44AAFF, "border_width": 0,
                            "margin": 0, "padding": 0, "font_style": "italic",
                            ":hover": {"color": 0xFF77CCFF}
                        })
                    ui.Spacer()

            ui.Spacer()


def on_manual_button_clicked(ext):
    abs_path = str(ext._manual_path.resolve())
    if not ext._manual_path.exists():
        print(f"[Error] Cannot find RAPID-RTX_User_Manual at: {abs_path}")
        return
    print(f"Opening User Manual: {abs_path}")
    webbrowser.open(f"file:///{abs_path}")


def on_github_button_clicked(ext):
    print(f"Opening GitHub: {ext._github_url}")
    webbrowser.open(ext._github_url)