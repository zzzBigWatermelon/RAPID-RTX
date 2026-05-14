import omni.ext
import omni.ui as ui
from omni.kit.menu.utils import MenuItemDescription, add_menu_items
import asyncio
import omni.kit.app
import carb.eventdispatcher
# 自定义模块
from .LiDAR import RTXLiDAR
from rapid.Utility import project_validity_check  # 项目有效性检查与文件路径获取
from rapid.Events import REQUEST_SIMULATION_START, SIMULATION_PARAMS_READY


class LiDARExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        print("[rapid.LiDAR] rapid LiDAR startup")
        self.__Lidar_simulation = RTXLiDAR()

        # 用于存储事件流回调的数据
        self._accumulated_data = {}
        self._observer = None

        # 定义一级菜单的名称
        self.TOP_MENU_NAME = "SimControl"
        # 定义菜单项
        self._menu_list = [
            MenuItemDescription(name="SimLiDAR", glyph="folder.svg", onclick_fn=self._on_button_clicked),]
        # 添加到顶部菜单栏
        asyncio.ensure_future(self._delayed_menu())

    async def _delayed_menu(self):
        '''异步等待menu菜单栏准备完毕后再添加'''
        # 等待Kit准备好
        for i in range(3):
            await omni.kit.app.get_app().next_update_async()
        add_menu_items(self._menu_list, self.TOP_MENU_NAME)

    def _on_button_clicked(self):
        # 检查项目文件环境完整性
        if not project_validity_check.get_current_usd_path() or not project_validity_check.quick_project_check():
            return
        # 清空旧数据和旧监听器
        self._accumulated_data = {}
        if self._observer:
            self._observer = None

        # 注册监听器
        self._observer = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=SIMULATION_PARAMS_READY,
            on_event=self._on_data_received,
            observer_name="rapid.simulation.ParamsReceiver")

        # 发出请求信号给其他扩展（如 Observation 扩展）
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            event_name=REQUEST_SIMULATION_START,
            payload={"sender": "LiDAR"})

    def _on_data_received(self, event: carb.eventdispatcher.Event):
        """收到参数后启动模拟
        """
        # 只接收对{"sender": "optical"}的回应
        if event.get("sender") != "LiDAR":
            return

        # 多个数据输入窗口的数据聚合,但是Lidar只用到Observation的数据
        print("[Rapid.LiDAR]收到一个事件包")
        # 尝试获取 Reflectance 数据
        if event.has_key("ReflectanceDatabase"):
            self._accumulated_data["ReflectanceDatabase"] = event.get("ReflectanceDatabase")
        # 尝试获取 Observation 数据
        if event.has_key("Observation"):
            self._accumulated_data["Observation"] = event.get("Observation")
            print("[Rapid.LiDAR]收到 Observation 数据")

        # 检查数据是否凑齐，齐全后开始模拟
        if "Observation" in self._accumulated_data and "ReflectanceDatabase" in self._accumulated_data:
            print("[LiDAR Sim] 所有数据已就绪，准备启动 LiDAR 模拟...")

            # 启动主程序
            self.__Lidar_simulation.main(self._accumulated_data)

            # 成功后清理
            self._observer = None
            self._accumulated_data = {}

    def on_shutdown(self):
        omni.kit.menu.utils.remove_menu_items(self._menu_list, self.TOP_MENU_NAME)
        self.__Lidar_simulation = None
        print("[rapid.LiDAR] rapid LiDAR shutdown")
