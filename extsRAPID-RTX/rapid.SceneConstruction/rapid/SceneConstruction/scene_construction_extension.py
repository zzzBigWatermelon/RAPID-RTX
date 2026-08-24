'''
底下很多的函数具体参数都没有指定,但是也没有报错,如show_window的value,_visiblity_changed_fn的visible,应该都是继承父类'''
from functools import partial
import asyncio
import omni.ext
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items, refresh_menu_items
import omni.ui as ui
import carb.eventdispatcher
from rapid.Events import REQUEST_SIMULATION_START, SIMULATION_PARAMS_READY
from .scene_construction_window import SceneConstructionWindow


class SceneConstructionExtension(omni.ext.IExt):
    '''这是Omniverse加载扩展的逻辑。
    1、当扩展启用时,将实例化从顶级模块中的“omni.ext.IExt”派生的任何类(在“extension.toml”的“python.modules”中定义)
       并调用“on_startup(ext_id)”。稍后当扩展被禁用时，将调用 on_shutdown()
    3、目前只需要更改WINDOW_NAME、MENU_PATH和引入的.window、窗口的停靠位置deferred_dock_in
    '''

    # The entry point for Scatter Window菜单的位置
    WINDOW_NAME = "Scene Construction"
    MENU_GROUP = "SimControl"

    def __init__(self):
        super().__init__()
        self._window = None
        self._observer = None
        self._cached_data = {}  # UI数据缓存
        self._menu_items = []

    def on_startup(self, ext_id):
        '''初始化函数，加载时自动调用
        ext_id 是当前扩展程序 ID。它可以与扩展管理器一起使用来查询其他信息，例如此扩展程序在文件系统上的位置。'''

        print("[rapid.SceneConstruction] rapid SceneConstruction startup")
        # The ability to show up the window if the system requires it. We use it in QuickLayout.
        # 注册一个函数，当窗口可见性发生变化时调用。
        ui.Workspace.set_show_window_fn(SceneConstructionExtension.WINDOW_NAME, partial(self.show_window, None))

        # 2. 使用 MenuItemDescription 添加菜单
        self._menu_items = [
            MenuItemDescription(
                name=self.WINDOW_NAME,
                ticked_fn=self._is_window_visible,
                onclick_fn=self._toggle_window
            )
        ]

        # 3. 添加到菜单栏
        add_menu_items(self._menu_items, self.MENU_GROUP)

        # 初始显示窗口
        self.show_window(None, True)

        # 监听 simulation.Start 请求事件
        self._observer = carb.eventdispatcher.get_eventdispatcher().observe_event(
                    event_name=REQUEST_SIMULATION_START,
                    on_event=self._on_start_request_received,
                    observer_name="rapid.observation.responderToSimualtionStart"
                )

    def _is_window_visible(self) -> bool:

        window = getattr(self, "_window", None)

        if window is None:
            return False

        return bool(window.visible)

    def _toggle_window(self):
        """点击菜单项时，反转窗口显示状态"""
        is_visible = self._is_window_visible()
        self.show_window(None, not is_visible)

    def show_window(self, menu, value):
        """控制窗口的显示/隐藏"""
        if value:
            if not self._window:
                self._window = SceneConstructionWindow(self.WINDOW_NAME, width=300, height=500)
                self._window.deferred_dock_in('Stage', ui.DockPolicy.DO_NOTHING)
                # 绑定窗口关闭事件（点击 X 号时同步菜单状态）
                self._window.set_visibility_changed_fn(self._visiblity_changed_fn)
            else:
                self._window.visible = True
        else:
            if self._window:
                self._window.visible = False

        # 状态改变后，手动刷新菜单 UI
        refresh_menu_items(self.MENU_GROUP)

    def _visiblity_changed_fn(self, visible):
        """窗口可见性发生变化的回调(例如用户点击了窗口右上角的 X)"""
        # 1. 刷新菜单勾选状态
        refresh_menu_items(self.MENU_GROUP)

        # 2. 如果不可见，则异步销毁窗口以节省内存
        # if not visible:
        #     asyncio.ensure_future(self._destroy_window_async())

    async def _destroy_window_async(self):
        """等待一帧后销毁窗口，这是 Kit 推荐的窗口清理方式"""
        await omni.kit.app.get_app().next_update_async()
        if self._window and not self._window.visible:
            self._window.destroy()
            self._window = None

    def on_shutdown(self):
        print("[rapid.AI] shutdown")

        # 移除菜单项
        if hasattr(self, '_menu_items') and self._menu_items:
            remove_menu_items(self._menu_items, self.MENU_GROUP)
            self._menu_items = []

        if self._window:
            self._window.destroy()
            self._window = None

        ui.Workspace.set_show_window_fn(self.WINDOW_NAME, None)
        self._observer = None

    def _on_start_request_received(self, event: carb.eventdispatcher.Event):
        """当收到 Simulation 的开始请求时调用"""
        # 数据请求方的识别sender_id
        sender = event.get("sender")

        current_data = {}
        # 2. 从UI获取当前数据
        if self._window:
            current_data = self._window.get_UI_data_to_simulation()
            self._cached_data = current_data  # 更新缓存

        # 3. 发送数据回对应的sender
        current_data = {"sender": sender, 'SceneConstruction': current_data}
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            event_name=SIMULATION_PARAMS_READY,
            payload=current_data
        )
        print(f"[ext: rapid.SceneConstruction] 数据已发送到rapid.Simulation sensor_stage_path: {current_data}")