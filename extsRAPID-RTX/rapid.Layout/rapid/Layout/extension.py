import omni.ext
import omni.ui as ui
from pathlib import Path
import json
import asyncio
import omni.kit.menu.utils as menu_utils
from omni.kit.menu.utils import MenuLayout, MenuItemDescription
import omni.kit.menu.utils.app_menu as app_menu_mod


class LayoutExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[ext:rapid.Layout] rapid Layout startup")
        # 针对报错的 _refresh_menu_item 函数进行加固
        # 无论它是模块函数还是类方法，我们都拦截它
        target = None
        # 尝试从模块中获取该函数
        if hasattr(app_menu_mod, "_refresh_menu_item"):
            target = app_menu_mod
        # 尝试从类中获取该函数 (Kit 106+ 常见)
        elif hasattr(app_menu_mod.AppMenu, "_refresh_menu_item"):
            target = app_menu_mod.AppMenu

        if target:
            orig_func = target._refresh_menu_item

            def safe_refresh(*args, **kwargs):
                try:
                    return orig_func(*args, **kwargs)
                except KeyError as e:
                    # 如果报错的是 Window_Browsers 或其他已移除的项，保持安静
                    if "Window_Browsers" in str(e) or "Window_Graph_Editors" in str(e):
                        return None
                    # 如果是其他真正的错误，还是抛出来
                    raise e

            # 替换系统函数
            target._refresh_menu_item = safe_refresh

        asyncio.ensure_future(self.layout())

    async def layout(self) -> None:
        """此函数将在app启动后加载UI窗口布局文件"""
        # 等待app启动完毕
        for i in range(3):
            await omni.kit.app.get_app().next_update_async()

        # 移除顶部菜单栏中多余的工具
        self.modify_menu_bar()
        # 这里加载布局文件,文件放在此扩展的data文件夹中
        layout_file_path = Path(__file__).parent.parent.parent/'data'/'Custom_Layout.json'
        # 读取布局文件并加载
        with open(layout_file_path, 'r') as file:
            json_string = file.read()
            data = json.loads(json_string)
        ui.Workspace.restore_workspace(data, True)

    def on_shutdown(self):
        print("ext:[rapid.SceneConstruction] rapid Layout shutdown")

    def modify_menu_bar(self):
        '''修改顶部菜单栏的原始布局,删除与RAPID不相关的内容'''
        # 假设你想把所有的系统菜单都重新排队
        menu_utils.add_menu_items([], "Window", 200)
        menu_utils.add_menu_items([], "SimControl", 500)
        menu_utils.add_menu_items([], "Tool", 900)   # 给 Tools 一个较大的位置
        menu_utils.add_menu_items([], "Support", 1000)  # 给 Support 一个最大的位置，确保它在最后
        # 定义布局
        _layout = [
            MenuLayout.Menu("File", [
                MenuLayout.Item("---------NEW---------"),
                MenuLayout.Item("New Project"),
                MenuLayout.Item("New"),
                MenuLayout.Item("Open"),
                MenuLayout.Item("Open Recent"),
                MenuLayout.Item("Re-open with New Edit Layer"),
                MenuLayout.Item("Exit"),
                MenuLayout.Item("---------SAVE---------"),
                MenuLayout.Item("Save"),
                MenuLayout.Item("Save As..."),
                MenuLayout.Item("Save With Options"),
                MenuLayout.Item("Save Flattened As..."),
                MenuLayout.Item("Collect and Save As..."),
                MenuLayout.Item("---------IMPORT---------"),
                MenuLayout.Item("Import"),
                MenuLayout.Item("Import from Onshape"),
                MenuLayout.Item("Export"),
                MenuLayout.Item("-------REFERENCE-------"),
                MenuLayout.Item("Add Reference"),
                MenuLayout.Item("Add Payload"),
                MenuLayout.Item("---------OTHER---------"),
                MenuLayout.Item("New From Stage Template"),


            ]),
            MenuLayout.Menu("Create", [
                MenuLayout.Menu("Create/Lights",
                                [MenuLayout.Item("Cylinder Light", remove=True),
                                 MenuLayout.Item("Disk Light", remove=True),
                                 MenuLayout.Item("Rect Light", remove=True),
                                 MenuLayout.Item("Sphere Light", remove=True)]),
                MenuLayout.Menu("Create/Materials", remove=True),
                MenuLayout.Menu("Create/Audio", remove=True),
                MenuLayout.Menu("Create/Graphs", remove=True),
                MenuLayout.Menu("Create/Physics", remove=True),
                MenuLayout.Menu("Create/Robots", remove=True),
                MenuLayout.Item("Camera", remove=True),
                MenuLayout.Item("Scope", remove=True),
                MenuLayout.Item("Xform", remove=True),
                MenuLayout.Item("April Tags", remove=True),
                MenuLayout.Menu("Create/Environments", remove=True),
                MenuLayout.Menu("Create/Sensors", remove=True),
            ]),
            MenuLayout.Menu("Window", [
                MenuLayout.Menu("Window/Browsers", remove=True),
                MenuLayout.Menu("Window/Examples", remove=True),
                MenuLayout.Menu("Window/Graph Editors", remove=True),
                MenuLayout.Item("Asset Validator", remove=True),
                MenuLayout.Item("Collection", remove=True),
                MenuLayout.Item("Hotkeys", remove=True),
                MenuLayout.Item("Physics Stage Settings", remove=True),
                MenuLayout.Item("Render Settings", remove=True),
                MenuLayout.Item("Scatter Window", remove=True)
            ]),
            MenuLayout.Menu("SimControl", [
                MenuLayout.Item("-----SIMULATION SETUP-----"),
                MenuLayout.Item("Observation"),
                MenuLayout.Item("Reflectance Database"),  # 确保名字与各个扩展中的 WINDOW_NAME 一致
                MenuLayout.Item("Scene Construction"),
                MenuLayout.Item("-----EXECUTION-----"),
                MenuLayout.Item("Start"),
                MenuLayout.Item("SimLiDAR"),
                MenuLayout.Item("Save Parameters"),
            ]),
            MenuLayout.Menu("Replicator", remove=True),
            MenuLayout.Menu("Tools", remove=True),
            MenuLayout.Menu("Utilities", remove=True),
            MenuLayout.Menu("Layouts", remove=True),
            MenuLayout.Menu("Help", remove=True),
            MenuLayout.Menu("Tool"),
            MenuLayout.Menu("Support"),
        ]
        # 应用布局
        menu_utils.add_layout(_layout)

        # 需要主动的添加分隔符项才能显示
        file_menu_list = [
            MenuItemDescription(name="-----SIMULATION SETUP-----", enabled=False),
            MenuItemDescription(name="---------NEW---------", enabled=False),
            MenuItemDescription(name="---------SAVE---------", enabled=False),
            MenuItemDescription(name="---------IMPORT---------", enabled=False),
            MenuItemDescription(name="-------REFERENCE-------", enabled=False),
            MenuItemDescription(name="---------OTHER---------", enabled=False),]
        # 添加到File菜单栏
        menu_utils.add_menu_items(file_menu_list, "File")

        # 刷新菜单
        menu_utils.rebuild_menus()
