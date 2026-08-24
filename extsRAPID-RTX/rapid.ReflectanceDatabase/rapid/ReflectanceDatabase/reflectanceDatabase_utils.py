
import omni.kit.app
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
import omni.kit.notification_manager as nm
# 自定义模块
from rapid.Utility.custom_json_encoder import CompactListEncoder
from rapid.Utility import project_validity_check


def parse_str_to_float_list(raw_str: str):
    """将 "0.2, 0.3" 这种字符串转为 [0.2, 0.3] 的通用辅助函数"""
    if not raw_str:
        return []
    try:
        # split(',') 按逗号分割, strip() 去除空格, if x.strip() 过滤空值
        return [float(x.strip()) for x in raw_str.split(',') if x.strip()]
    except ValueError:
        print(f"[Warning] 无法解析数值: {raw_str}")
        return []


def parse_UI_data_to_dic(ref_tra_data_model, bands_data_model):
    '''将反射率表格UI中的数据解析为字典'''
    # 初始化结果字典
    result_data = {}
    ref_tra_data_data = {}

    # 解析遍历反射率数据表格UI 模型 items
    for item in ref_tra_data_model._items:
        # 获取名字 (Key)
        name = item.name_model.as_string

        # 获取原始字符串
        ref_raw = item.ref_value_model.as_string  # 例如 "0.3" 或 "0.2,0.3,0.4"
        tra_raw = item.tra_value_model.as_string
        display_color_raw = item.display_color_model.as_string

        # 构建字典结构
        ref_tra_data_data[name] = {
            'ref': parse_str_to_float_list(ref_raw),
            'tra': parse_str_to_float_list(tra_raw),
            'display_color': parse_str_to_float_list(display_color_raw)
        }

    # 解析波长数据
    bands_data = bands_data_model.as_string
    # 建立数据字典
    result_data['bands_data'] = bands_data
    result_data['ref_tra_data'] = ref_tra_data_data
    # 结果为{'leaf': {'ref': [0.3], 'tra': [0.3]}, 'Name': {'ref': [0.2, 0.3, 0.4], 'tra': [0.2, 0.3, 0.4]}}
    return result_data


def save_UI_data_to_json(bands_model, data_table_model):
    """
    保存当前波段数据,反射率表数据到 parameters json 文件中
    :param data: 要保存的反射率表格UI model
    """
    # 反射率表格UI数据解析为字典格式
    UI_data = parse_UI_data_to_dic(data_table_model, bands_model)

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
    existing_data["ReflectanceDatabase_bands"] = UI_data['bands_data']
    existing_data["ReflectanceDatabase"] = UI_data['ref_tra_data']

    # 保存新数据
    simulation_parameters_file.parent.mkdir(parents=True, exist_ok=True)
    with open(simulation_parameters_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, cls=CompactListEncoder, indent=4, ensure_ascii=False)


def read_csv_file(csv_path):
    '''读取csv文件数据,以逗号分割数据,第一行是数据名, 格式如下：
    wavelength(nm),reflectance(%),diffuse_transmittance(%)
    350,6.186,0....
    参数:
        csv_path:光谱数据文件的路径，非参数文件路径

    返回:
        refl: 反射率数据列表
        trans:透射率数据列表
        wavelength:波长数据列表
    '''
    # 读取数据
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # 取波长列
    wavelength = (df.iloc[:, 0].astype(float).values).tolist()
    # 直接取第2列反射率数据(索引为1)
    refl = (df.iloc[:, 1].astype(float).values).tolist()
    # 透射率 (第3列) - 增加判断防止有的CSV只有两列
    if df.shape[1] > 2:
        trans = (df.iloc[:, 2].astype(float).values).tolist()
    else:
        trans = [0.0] * len(refl)  # 如果没有透射率,补0
    return refl, trans, wavelength


def calculate_band_mean_value(wavelength, spectrum_values, center_wl, bandwidth):
    """
    辅助函数：计算指定中心波长和带宽内的平均值。
    参数:
        wavelength: 波长列表/数组 (例如[400, 401, 402...])
        spectrum_values: 对应的反射率/透射率列表/数组 (例如[0.1, 0.15, 0.12...])
        center_wl: 目标中心波长
        bandwidth: 目标波段宽度
    """
    min_wl = center_wl - (bandwidth / 2.0)
    max_wl = center_wl + (bandwidth / 2.0)

    valid_values = []
    closest_val = 0.0
    min_diff = float('inf')

    # 使用 zip() 将波长和对应的数据打包，同时遍历
    for wl_val, spec_val in zip(wavelength, spectrum_values):
        wl = float(wl_val)
        val = float(spec_val)

        # 1. 寻找落在波段区间内的所有点
        if min_wl <= wl <= max_wl:
            valid_values.append(val)

        # 2. 同时记录距离中心波长最近的点 (以备后用)
        diff = abs(wl - center_wl)
        if diff < min_diff:
            min_diff = diff
            closest_val = val

    # 如果区间内找到了数据点，计算平均值
    if valid_values:
        return sum(valid_values) / len(valid_values)
    else:
        # 如果波段极窄 (带宽为0) 或该区间无采样点，直接返回最接近中心波长的数值
        return closest_val


def get_spectrum_data_for_bands(spectra_file_path, bands_str):
    """
    根据输入的 bands_str "400:1,500:1..." 和光谱数据，
    返回格式化后的 (ref_str, tra_str)
    参数:
        parameters_file_path:光谱数据文件的路径，非参数文件路径
        bands_str: "400:1,500:1...", 波段数据形如波长:波段宽度,....., 从UI数据中得到
    """
    refl, trans, wavelengths = read_csv_file(spectra_file_path)

    ref_results = []
    tra_results = []

    band_parts = bands_str.split(',')
    for bp in band_parts:
        if not bp.strip() or ':' not in bp:
            continue
        # 按照冒号分割中心波长和波段宽度
        center_wl, bandwidth = map(float, bp.split(':'))
        # 计算该波段内的反射率和透射率
        ref_val = calculate_band_mean_value(wavelengths, refl, center_wl, bandwidth)
        tra_val = calculate_band_mean_value(wavelengths, trans, center_wl, bandwidth)
        # 保留三位小数
        ref_results.append(f"{ref_val:.4f}")
        tra_results.append(f"{tra_val:.4f}")
    # 5格式化拼接成 "0.200,0.300,0.400" 这种字符串
    ref_str = ",".join(ref_results)
    tra_str = ",".join(tra_results)

    # 添加到反射率UI表格中
    return ref_str, tra_str


def read_spectra_file_path(default_parameters_path, project_parameters_path):
    """扫描默认库和项目库中的 simulation_parameters.json, 读取其中的光谱文件路径
    参数:
        default_parameters_path (str):
        project_parameters_path (str):"""
    config_paths = [default_parameters_path, project_parameters_path]
    spectra_database = {}
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                # 读取parameters.json文件，后面获取spectra_database文件名字
                data = json.load(f)
                # 获取当前 JSON文件/spectra_database 所在的根文件夹
                base_dir = Path(config_path).parent

                for item in data.get("spectra_data_info", []):
                    # 获取spectra_database文件路径
                    # 拼接并检查 .csv 文件的物理存在性
                    csv_path = (base_dir / "spectra_database" / item["file"]).resolve()

                    if csv_path.exists():
                        # 核心逻辑：如果文件存在，则写入字典。
                        # 如果项目库中有同名 item['name']，它会在此处自动覆盖默认库的路径。
                        spectra_database[item['name']] = {
                            "csv_path": csv_path.as_posix(),
                            "metadata": item  # 其他信息
                        }
    return spectra_database


def parse_json_to_UI_format(file_path):
    '''解析 simulation_parameters.JSON 数据的辅助函数
    参数:
        default_parameters_path (str):simulation_parameters.json文件的路径
    返回:
        bands_str (str): 波段定义的数据,'波长:波段宽度，.....'
        new_rows List[List[name, ref, tra, display]]:反射率表格每一行的参数
        '''
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bands_str = data.get("ReflectanceDatabase_bands", '')
    ref_dict = data.get("ReflectanceDatabase", {})
    new_rows = []
    for mat_name, mat_props in ref_dict.items():
        # 使用列表推导式高效处理字段，默认值为 []
        # 统一处理 ref, tra, display_color
        fields = []
        for key in ["ref", "tra", "display_color"]:
            val_list = mat_props.get(key, [])
            fields.append(",".join(map(str, val_list)))
        new_rows.append([mat_name, *fields])
    return bands_str, new_rows
