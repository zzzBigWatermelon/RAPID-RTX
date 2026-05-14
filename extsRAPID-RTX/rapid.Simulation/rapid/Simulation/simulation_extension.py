'''
以下是后续应该写在readme中的内容
这是用于控制仿真进程的位于菜单栏的按钮,
这个代码仿照LAYOUT CONTROL TOOLS(omni.kit.quicklayout = { version = "1.0.7" })的代码的逻辑

'''
import os
import omni.ext
import omni.kit.actions.core
from omni.kit.menu.utils import MenuItemDescription, add_menu_items
import carb.eventdispatcher
import json
from .simulation import Simulation
# 自定义模块
from rapid.Events import REQUEST_SIMULATION_START, SIMULATION_PARAMS_READY
from rapid.Utility.custom_json_encoder import CompactListEncoder  # json文件的编码格式
from rapid.Utility import project_validity_check  # 项目有效性检查与文件路径获取


class SimulationExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        '''注册菜单按钮
        '''

        # 初始化
        self.__simulation = Simulation()
        self._observer = None

        # 用于事件流多次回调_on_data_received的数据聚合,暂存碎片数据
        self._accumulated_data = {}
        # 状态标记: True 表示需要模拟, False 表示仅保存
        self._is_simulation_mode = True

        # 定义一级菜单的名称
        self.TOP_MENU_NAME = "SimControl"
        # 定义菜单项
        self._menu_list = [
            MenuItemDescription(name="-----EXECUTION-----", enabled=False),
            MenuItemDescription(name="Start", glyph="folder.svg", onclick_fn=self._on_start_button_clicked),
            MenuItemDescription(name="Save Parameters", glyph="folder.svg", onclick_fn=self._on_save_button_clicked),]

        # 添加到顶部菜单栏
        add_menu_items(self._menu_list, self.TOP_MENU_NAME)

    def _on_start_button_clicked(self):
        '''工作流 1:请求 -> 保存 -> 模拟'''
        carb.log_info("[exts rapid.Simulation] Save and start the simulation")
        self._is_simulation_mode = True
        self._trigger_data_request()

    def _on_save_button_clicked(self):
        '''工作流 2:请求 -> 保存 -> 停止'''
        carb.log_info("[exts rapid.Simulation] Request data and save only")
        self._is_simulation_mode = False
        self._trigger_data_request()

    def _trigger_data_request(self):
        '''start和save parameters按钮通用数据请求流程
        第一步:建立监听器observe_event,监听其他扩展(如rapid.observation)传来的模拟参数
        第二步:发出请求dispatch_event,其他扩展监听到请求时发送模拟参数被监听器observe_event接受
        '''
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
            payload={"sender": "optical"})

    def _on_data_received(self, event: carb.eventdispatcher.Event):
        """收到参数后启动模拟
        注意:一、carb.events在Isaac Sim 4.5(基于Kit 106,但是改用Events 2.0)
        Events 1.0 (carb.events): 事件对象有一个.payload 属性，里面是字典,通过e.payload["boolParam"]获取参数。
        Events 2.0 (carb.eventdispatcher): 事件对象本身就是字典，没有.payload 属性,直接e["boolParam"]获取参数。
        二、这个函数会被调用多次：
        第1次可能是 Observation 发回的数据,第2次可能是 Reflectance 发回的数据,顺序是不确定的(竞态)，这里要合并数据。
        参数:
            self._accumulated_data (dict):数据格式如下{
            'Observation': {'sensor_stage_path': '/World/sensor'},
            'ReflectanceDatabase': {'leaf': {'ref': [0.2, 0.3, 0.4,...], 'tra': [0.2, 0.3, 0.4,...]}, 'Name': {'ref': [0.2, 0.3, 0.4,...], 'tra': [0.2, 0.3, 0.4,...]}}
            }
        """
        # 只接收对{"sender": "optical"}的回应,因为还有激光雷达的模拟回调，加上sender判断防止冲突
        if event.get("sender") != "optical":
            return

        # 多个扩展的数据聚合
        # 尝试获取 Reflectance 数据
        if event.has_key("ReflectanceDatabase"):
            self._accumulated_data["ReflectanceDatabase"] = event.get("ReflectanceDatabase")
            carb.log_info("[ext: rapid.Simulation] Received data from [ext: rapid.ReflectanceDatabase]")

        # 尝试获取 Observation 数据
        if event.has_key("Observation"):
            self._accumulated_data["Observation"] = event.get("Observation")
            carb.log_info("[ext: rapid.Simulation]Received data from[ext: rapid.Observation]")

        # 尝试获取 SceneConstruction 数据
        if event.has_key("SceneConstruction"):
            self._accumulated_data["SceneConstruction"] = event.get("SceneConstruction")
            carb.log_info("[ext: rapid.Simulation]Received data from[ext: rapid.SceneConstruction]")

        # 检查数据是否凑齐，齐全后开始模拟
        has_all_data = "ReflectanceDatabase" in self._accumulated_data and "Observation" in self._accumulated_data and "SceneConstruction" in self._accumulated_data

        # 开始执行模拟或者保存
        if has_all_data:
            # 先保存一次数据
            # self._save_simulation_data()

            # 启动模拟
            if self._is_simulation_mode:
                carb.log_info("[ext: rapid.Simulation] Starting simulation...")
                self.__simulation.start(self._accumulated_data)
            else:
                carb.log_info("[ext: rapid.Simulation] Data has been saved; simulation will not be started.")

            # 销毁监听器
            self._observer = None
        else:
            carb.log_info("[ext: rapid.Simulation] The data is incomplete; awaiting the next event data transmission....")

    def _save_simulation_data(self):

        # 这里首先检查项目文件的完整性
        current_usd_parent_dir = project_validity_check.get_current_usd_path()
        if not current_usd_parent_dir:
            return None
        if not project_validity_check.quick_project_check():
            return None

        # 获取参数文件夹
        parameters_dir = project_validity_check.get_folder("parameters")
        save_path = os.path.join(parameters_dir, 'simulation_parameters.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._accumulated_data, f, ensure_ascii=False, indent=4, cls=CompactListEncoder)

    def on_shutdown(self):  # pragma: no cover
        omni.kit.menu.utils.remove_menu_items(self._menu_list, self.TOP_MENU_NAME)

        action_registry = omni.kit.actions.core.get_action_registry()
        if action_registry:
            action_registry.deregister_all_actions_for_extension("rapid.Simulation")
        self.__simulation = None
