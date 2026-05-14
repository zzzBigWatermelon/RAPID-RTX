import json
import omni.ui as ui
from pathlib import Path
import carb  # 必须导入 carb 模块
# 自定义模块
from rapid.Utility.custom_json_encoder import CompactListEncoder
from rapid.Utility import project_validity_check

# 定义文件路径
DATA_ROOT = Path(__file__).parent.parent.parent/'data'


class SceneConstructionUtils:

    @staticmethod
    def get_ui_value(UI_model):
        """根据模型类型自动提取值"""
        # 1. 处理列表情况 (如 XYZ 坐标)
        if isinstance(UI_model, list):
            return [SceneConstructionUtils.get_ui_value(m) for m in UI_model]  # 递归提取列表里的每个模型

        # 2. 处理 ComboBoxModel
        if hasattr(UI_model, "get_current_item"):
            item = UI_model.get_current_item()
            if isinstance(UI_model, ui.SimpleStringModel):
                return item.as_string
            return item.as_string

        # 3. 严格类型判断 (解决 0.0 问题的关键)
        # 必须先判断 String 类型，或者通过类型名判断
        if isinstance(UI_model, ui.SimpleStringModel):
            return UI_model.as_string
        if isinstance(UI_model, ui.SimpleFloatModel):
            return UI_model.as_float
        if isinstance(UI_model, ui.SimpleIntModel):
            return UI_model.as_int
        return UI_model

    @staticmethod
    def save_UI_data_to_json(UI_model):
        """
        保存当前窗口数据到simulation_parameters.json文件中
        :param data: 要保存的反射率表格UI model
        """
        # 遍历全部模型并提取值
        data = {}
        for key, model in UI_model.items():
            raw_value = SceneConstructionUtils.get_ui_value(model)  # 调用提取函数
            if key == "omnidirectional_sampling_view_zenith":
                # 从str类型中拆分出float类型数据
                data[key] = [float(x.strip()) for x in raw_value.split(',') if x.strip()]
            else:
                data[key] = raw_value

        # 检查项目文件环境完整性
        if not project_validity_check.get_current_usd_path() or not project_validity_check.quick_project_check():
            return
        # 获取参数文件路径
        parameters_path = Path(project_validity_check.get_folder("parameters"))
        simulation_parameters_file = parameters_path / 'simulation_parameters.json'

        existing_data = {}
        # 先读取之前文件中的旧数据 (增加文件是否存在的判断以防报错)
        with open(simulation_parameters_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        # 覆盖旧数据
        existing_data["SceneConstruction"] = data

        # 保存新数据
        simulation_parameters_file.parent.mkdir(parents=True, exist_ok=True)
        with open(simulation_parameters_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, cls=CompactListEncoder, indent=4, ensure_ascii=False)

    @staticmethod
    def read_json_data_to_UI(UI_model_dict):
        """
        从 simulation_parameters.json 读取数据并自动填充到 self.models 中
        :param UI_model_dict: 你的 self.models 字典
        """
        # -----------------------获取模拟参数文件路径---------------------
        # 项目的模拟参数文件路径
        parameters_path = Path(project_validity_check.get_folder("parameters"))
        simulation_parameters_file = parameters_path / 'simulation_parameters.json'

        # 如果没有项目，就是用默认参数
        if not simulation_parameters_file.exists():
            default_parameters_file = DATA_ROOT / 'simulation_parameters.json'
            if default_parameters_file.exists():
                simulation_parameters_file = default_parameters_file
                carb.log_info(f"[ext: rapid.SceneConstruction] The extension's default parameters are loaded.: {simulation_parameters_file}")
            else:
                carb.log_warn("[ext: rapid.SceneConstruction] No configuration files (simulation_parameters.json) found.")
                return

        # -----------------------读取 JSON 文件---------------------
        try:
            with open(simulation_parameters_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                obs_data = json_data.get("SceneConstruction", {})
        except Exception as e:
            carb.log_warn(f"[ext: rapid.SceneConstruction] Failed to parse JSON file: {e}")
            return

        # 3. 自动遍历并下发数据
        for key, value in obs_data.items():
            if key not in UI_model_dict:
                continue  # 如果 JSON 中有多余的键，忽略

            ui_model = UI_model_dict[key]

            # --- A. 处理列表类型 (如 [ui.SimpleFloatModel, ui.SimpleFloatModel, ...]) ---
            if isinstance(ui_model, list):
                if isinstance(value, list):
                    # 使用 zip 同时遍历 UI 列表和 JSON 数值列表，非常高效
                    for sub_model, sub_val in zip(ui_model, value):
                        if isinstance(sub_model, ui.SimpleFloatModel):
                            sub_model.as_float = float(sub_val)
                        elif isinstance(sub_model, ui.SimpleIntModel):
                            sub_model.as_int = int(sub_val)

            # --- C. 处理 ComboBoxModel (假设它的内部实现有匹配字符串的机制) ---
            elif hasattr(ui_model, "get_item_value_model"): 
                # 通常自定义 ComboBoxModel 拥有选项列表，我们需要找到匹配的字符串并设置 Index
                # 假设你的 ComboBoxModel 暴露了 items 列表：
                if hasattr(ui_model, "items"):
                    for i, item_str in enumerate(ui_model.items):
                        if item_str == value:
                            ui_model.get_item_value_model().as_int = i
                            break

            # --- D. 处理标准原子类型 ---
            elif isinstance(ui_model, ui.SimpleStringModel):
                ui_model.as_string = str(value)
            elif isinstance(ui_model, ui.SimpleFloatModel):
                ui_model.as_float = float(value)
            elif isinstance(ui_model, ui.SimpleIntModel):
                ui_model.as_int = int(value)

        carb.log_info("[ext: rapid.SceneConstruction] The JSON parameters have been successfully synchronized to the UI")