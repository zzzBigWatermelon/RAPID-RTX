'''2025.11.12
重新说一下对mcv逻辑的理解
'''
from functools import partial
import asyncio
import omni.ext
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items, refresh_menu_items
import omni.ui as ui
import carb.eventdispatcher
# 自定义模块
from .reflectanceDatabase_window import ReflectanceDatabaseWindow
from rapid.Events import REQUEST_SIMULATION_START, SIMULATION_PARAMS_READY
from .reflectanceDatabase_utils import parse_UI_data_to_dic


class ReflectanceDatabaseExtension(omni.ext.IExt):
    '''早前的注释:这是Omniverse加载扩展的逻辑。
    1、当扩展启用时,将实例化从顶级模块中的“omni.ext.IExt”派生的任何类(在“extension.toml”的“python.modules”中定义)
       并调用“on_startup(ext_id)”。稍后当扩展被禁用时，将调用 on_shutdown()
    2、底下很多的函数调用时没有指定参数,但是也没有报错,如show_window的value(通过ui.Workspace.show_window调用),
       _visiblity_changed_fn的visible,应该都是继承父类，但好像都是控制菜单加载的，不重要
    3、目前只需要更改WINDOW_NAME、MENU_PATH和引入的.window、窗口的停靠位置deferred_dock_in
    '''

    # The entry point for Scatter Window菜单的位置
    WINDOW_NAME = "Reflectance Database"
    MENU_GROUP = "SimControl"

    def __init__(self):
        super().__init__()
        self._window = None
        self._observer = None
        self._cached_data = {}
        self._menu_items = []

    def on_startup(self, ext_id):
        '''初始化函数，加载时自动调用
        ext_id 是当前扩展程序 ID。它可以与扩展管理器一起使用来查询其他信息, 例如此扩展程序在文件系统上的位置。'''

        print("[rapid.ReflectanceDatabase] rapid ReflectanceDatabase startup")
        # The ability to show up the window if the system requires it. We use it in QuickLayout.
        # 注册一个函数，当窗口可见性发生变化时调用。
        ui.Workspace.set_show_window_fn(ReflectanceDatabaseExtension.WINDOW_NAME, partial(self.show_window, None))

        # 2. 使用 MenuItemDescription 添加菜单
        self._menu_items = [
            MenuItemDescription(
                name=self.WINDOW_NAME,
                ticked_fn=self._is_window_visible,
                onclick_fn=self._toggle_window
            )
        ]

        # 3. 添加到菜单栏 (注意：第一个参数必须是 list)
        add_menu_items(self._menu_items, self.MENU_GROUP)

        # 初始显示窗口
        self.show_window(None, True)

        # 监听 simulation.Start 请求事件
        self._observer = carb.eventdispatcher.get_eventdispatcher().observe_event(
                    event_name=REQUEST_SIMULATION_START,
                    on_event=self._on_start_request_received,
                    observer_name="rapid.observation.responderToSimualtionStart"
                )

    def _on_start_request_received(self, event: carb.eventdispatcher.Event):
        """当收到 Simulation 的开始请求时调用"""
        print("[Rapid.ReflectanceDatabase] 收到rapid.simulation开始请求,正在读取 UI 数据...")
        # 数据请求方的识别sender_id
        sender = event.get("sender")

        current_data = {}
        # 2. 从 UI 获取当前数据
        if self._window:
            try:
                current_data = parse_UI_data_to_dic(self._window.ref_data_model, self._window.bands_data_model)
                self._cached_data = current_data  # 更新缓存
            except Exception as e:
                carb.log_error(f"Error getting data from window: {e}")
        else:
            print("[Rapid.ReflectanceDatabase] 窗口已关闭，使用最后一次缓存的数据")
            current_data = self._cached_data

        # 3. 发送数据回Simulation
        current_data = {"sender": sender, 'ReflectanceDatabase': current_data}
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            event_name=SIMULATION_PARAMS_READY,
            payload=current_data
        )
        print(f"[Rapid.ReflectanceDatabase] 数据已发送到rapid.Simulation current_data: {current_data}")

    def _is_window_visible(self) -> bool:
        """返回窗口当前的可见状态，用于给菜单栏打钩"""
        return self._window is not None and self._window.visible

    def _toggle_window(self):
        """点击菜单项时，反转窗口显示状态"""
        is_visible = self._is_window_visible()
        self.show_window(None, not is_visible)

    def show_window(self, menu, value):
        """控制窗口的显示/隐藏"""
        if value:
            if not self._window:
                self._window = ReflectanceDatabaseWindow(self.WINDOW_NAME, width=300, height=500)
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
        print("[rapid.Observation] shutdown")

        # 移除菜单项
        if hasattr(self, '_menu_items') and self._menu_items:
            remove_menu_items(self._menu_items, self.MENU_GROUP)
            self._menu_items = []

        if self._window:
            self._window.destroy()
            self._window = None

        ui.Workspace.set_show_window_fn(self.WINDOW_NAME, None)
        self._observer = None
