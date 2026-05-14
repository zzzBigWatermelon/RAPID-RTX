import os
import re
import numpy as np
# from .radiometric_calibration import RadiometricCalibration
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from pathlib import Path


class NpyToReflectanceHdrConverter:
    '''
    '''
    def __init__(self, intermediate_path, result_path):
        """
        intermediate_path:临时缓存文件base_dir
        result_path:处理完成的结果文件夹
        """
        self.intermediate_path = str(intermediate_path)
        self.result_path = str(result_path)

    def npy_to_multichannel_tiff(self, models, bands_str):
        """
        主函数：将每个 pos 对应的多个 npy 文件合成为多通道TIFF
        并根据 bands_str 限制最大波段数，同时写入波长元数据。
        """
        # 解析波段字符串
        wavelengths, total_bands = self.parse_bands_string(bands_str)

        # 获取所有 npy 文件名
        all_files = self.discover_files(self.intermediate_path, 'HDR_pos')

        # 按位置pos分组
        pos_files = self.group_files_by_pos(all_files)

        # 逐位置处理
        for pos_str, file_list in pos_files.items():
            self.process_single_pos(pos_str, file_list, self.intermediate_path, models, wavelengths, total_bands, self.result_path)

    def parse_bands_string(self, bands_str):
        """
        解析波段字符串，提取波长列表。
        参数
            bands_str (str): "800.00:1.0, 844.44:1.0, 888.89:1.0"
        返回:
            波长列表 List:wavelengths_list = ['800.00', '844.44', '888.89']
            波长总数 (int):total_bands
        """
        wavelengths = []
        for part in bands_str.split(','):
            part = part.strip()
            if ':' in part:
                wl = part.split(':')[0].strip()
            else:
                wl = part
            wavelengths.append(wl)
        return wavelengths, len(wavelengths)

    @staticmethod
    def natural_key(s: str):
        '''排序文件名'''
        return [int(chunk) if chunk.isdigit() else chunk.lower()
                for chunk in re.split(r'(\d+)', s)]

    def discover_files(self, data_files_path, data_type='HDR_radiometric_calibration'):
        '''
        '''
        image_files = [
            f for f in os.listdir(data_files_path)
            if f.lower().endswith('.npy') and f.startswith(data_type)
        ]
        return sorted(image_files, key=NpyToReflectanceHdrConverter.natural_key)

    def group_files_by_pos(self, file_list):
        """
        从文件列表中提取 pos 编号，并按 pos 分组。
        输入:
            file_list List[str]: file_list = ['pos0001_band0001.npy', 'pos0001_band0002.npy', 'pos0002_band0001.npy']
        输出: 
            每个位置pos对应的所有文件名 Dict:{'1': ['pos1_band1.npy', 'pos1_band2.npy'], '2': ['pos2_band1.npy']}
        """
        pos_files = {}
        for fname in file_list:
            match = re.search(r"pos(\d+)", fname, re.IGNORECASE)
            if not match:
                continue
            pos_str = match.group(1)
            pos_files.setdefault(pos_str, []).append(fname)
        # 对每个 pos 的文件列表排序，保证顺序稳定
        for pos_str in pos_files:
            pos_files[pos_str].sort()
        return pos_files

    def process_single_pos(self, pos_str, file_list, intermediate_data_path, models, wavelengths, total_bands, result_path):
        """
        处理一个 pos 的所有文件，按顺序读取每个文件的 R,G,B 通道，
        转换为反射率，直至达到 total_bands 为止，然后保存为多通道 TIFF。
        返回: (success, actual_bands)
        """
        channel_keys = ['R', 'G', 'B']
        all_bands = []       # 存储每个通道的反射率数据 (H,W)
        bands_added = 0      # 已添加的通道数

        for fname in file_list:
            # --- 硬切断：超过总波段数则不处理 ---
            if bands_added >= total_bands:
                break

            data_path = os.path.join(intermediate_data_path, fname)
            raw_RGBA = np.load(data_path)
            raw_data = raw_RGBA[:, :, :3].astype(np.float32)   # 取前三个通道

            for i, key in enumerate(channel_keys):
                # --- 硬切断：超过总波段数则不处理 ---
                if bands_added >= total_bands:
                    break
                band_data = raw_data[..., i]
                slope = models[key]['slope']
                intercept = models[key]['intercept']

                if abs(slope) < 1e-10:
                    reflectance = np.zeros_like(band_data)
                else:
                    reflectance = (band_data - intercept) / slope

                all_bands.append(reflectance)
                bands_added += 1

        # 堆叠（C 为通道数，H 为高度，W 为宽度）
        stacked = np.stack(all_bands, axis=0)  # (C, H, W)
        data_cube = np.transpose(stacked, (1, 2, 0))  # 变为 (H, W, C)
        # 实际写入到 ENVI 文件中的波段对应的波长列表
        used_wavelengths = wavelengths[:bands_added]
        # 输出路径
        output_path_base = os.path.join(result_path, f"pos{pos_str}_multiband")
        # 准备ENVI头文件元数据
        metadata = {
            'description': 'Reflectance data converted from simulation',
            'wavelength units': 'nm',
            'wavelength': used_wavelengths,
        }
        # 直接保存（自动处理 interleave 等）
        envi.save_image(output_path_base + '.hdr', data_cube, ext='.dat', dtype=np.float32,
                        interleave='bsq', metadata=metadata, force=True)
        return True, bands_added


class BRFCurveCalculation:
    '''
    '''
    @staticmethod
    def main(hdr_folder_path='', zenith_range=[60, -60], zenith_step=5):
        '''主流程：读取所有 .hdr 文件，计算每个波段的平均反射率，按角度排序后输出字典。

        参数:
            hdr_file_path (str): hdr数据文件夹路径
            zenith_range (List): 天顶角的模拟范围,只包含起始和结束天顶角数值的列表
            zenith_step (int): 天顶角的模拟步长

        '''
        # 读取所有文件名
        all_files = BRFCurveCalculation.discover_files(hdr_folder_path, 'pos', '.hdr')
        if not all_files:
            raise FileNotFoundError(f"No files were found.{hdr_folder_path} ")

        #  生成角度序列（自动判断方向）
        start_angle, end_angle = zenith_range[0], zenith_range[1]
        actual_step = -abs(zenith_step) if start_angle > end_angle else abs(zenith_step)
        # 生成角度序列
        angles = np.arange(start_angle, end_angle, actual_step)
        # 确保角度数量与文件数量一致（若不一致会警告，但继续处理）
        if len(angles) != len(all_files):
            print(f": The length of the angle sequence ({len(angles)}) is inconsistent with the number of files ({len(all_files)})")

        # 读取第一个文件以获取波段数量及波长,初始化结果字典：{波长: [平均值列表]}
        first_file = os.path.join(hdr_folder_path, all_files[0])
        _, wavelengths, _ = BRFCurveCalculation.read_hdr_file_mean_bands(first_file)
        result_by_wavelength = {wl: [] for wl in wavelengths}

        # 遍历每个文件,计算所有角度的平均值
        for idx, filename in enumerate(all_files):
            file_path = os.path.join(hdr_folder_path, filename)
            band_means, _, _ = BRFCurveCalculation.read_hdr_file_mean_bands(file_path)
            # 将当前文件各波段的平均值添加到对应的列表中
            for idx, wl in enumerate(wavelengths):
                result_by_wavelength[wl].append(band_means[idx])

        # 画BRF图像
        BRFCurveCalculation.plot_brf_curves(result_by_wavelength, angles, hdr_folder_path)

    @staticmethod
    def natural_key(s: str):
        '''排序文件名'''
        return [int(chunk) if chunk.isdigit() else chunk.lower()
                for chunk in re.split(r'(\d+)', s)]

    def discover_files(data_files_path, file_prefix='HDR_radiometric_calibration',
                       file_suffix='.hdr'):
        '''发现文件夹下所有符合前缀和后缀的文件，并按自然顺序排序
        '''
        image_files = [
            f for f in os.listdir(data_files_path)
            if f.lower().endswith(file_suffix) and f.startswith(file_prefix)
        ]
        return sorted(image_files, key=BRFCurveCalculation.natural_key)

    @staticmethod
    def read_hdr_file_mean_bands(hdr_file_path: str):
        """
        读取单个 ENVI 文件(.hdr),计算每个波段的整体平均值(所有像素的均值).
        返回:
            band_means: numpy.ndarray, shape (n_bands,)，每个波段的平均值
            wavelengths: numpy.ndarray, shape (n_bands,)，波长列表（单位由元数据决定）
        """
        img = envi.open(hdr_file_path)
        data_cube = img.load()          # 形状: (rows, cols, bands)
        # 计算每个波段的所有像素均值
        # axis=(0,1) 表示对前两维（行和列）求平均
        band_means = np.mean(data_cube, axis=(0, 1))  # 长度 = bands
        wavelengths = np.array(img.bands.centers, dtype=np.float32)  # 波长列表List
        wavelength_units = img.bands.band_unit  # 波长单位
        return band_means, wavelengths, wavelength_units

    @staticmethod
    def plot_brf_curves(result_by_wavelength, angles, hdr_folder_path, title="BRF Curves by Wavelength",
                        xlabel="Viewn Zenith Angle (degree)", ylabel="Reflectance"):
        """
        绘制多波长 BRF 曲线（反射率随观测角度变化）

        参数:
            result_by_wavelength: dict, 键为波长 (float), 值为该波长在各角度下的反射率列表 (list of float)
            angles: list or array, 观测角度序列，长度应与反射率列表长度一致
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
        """
        plt.figure(figsize=(10, 6))

        # 对波长进行排序，以便图例有序
        sorted_wavelengths = sorted(result_by_wavelength.keys())

        for wl in sorted_wavelengths:
            reflectance = result_by_wavelength[wl]
            # 确保角度与反射率数据长度一致（如果角度是反序，可以在这里处理，但假定已对齐）
            plt.plot(angles, reflectance, marker='o', linewidth=1.5, label=f"{wl:.1f} nm")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Wavelength")
        plt.tight_layout()
        save_path = Path(hdr_folder_path) / 'BRF_result.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


# 测试保留
if __name__ == "__main__":
    # ------------测试NpyToReflectanceHdrConverter------------------
    # # 反射板信息(要与场景中的虚拟反射板信息一致)
    # REFLECTANCE_VALUES = [0.03, 0.05, 0.07]
    # REFLECTANCE_KEYS = ['3%', '5%', '7%']
    # reflectance_panel_Semantic_Type = 'reflectance_panel'
    # reflectance_panel_Semantic_Data = ['reflectance_panel_3', 'reflectance_panel_5', 'reflectance_panel_7']
    # base_dir = r'C:\Users\ZZZ\Desktop\22222222\intermediate_data'
    # result_dir = r'C:\Users\ZZZ\Desktop\22222222\result'
    # # 处理流程
    # # 如果报错没有RadiometricCalibration，就去上边切换RadiometricCalibration导入方式
    # rc_pipeline = RadiometricCalibration(base_dir=base_dir)
    # models = rc_pipeline.radiometric_calibration_pipeline(REFLECTANCE_VALUES, REFLECTANCE_KEYS, reflectance_panel_Semantic_Type, reflectance_panel_Semantic_Data)
    # npytotiff_pipeline = NpyToReflectanceHdrConverter(base_dir, result_dir)
    # npytotiff_pipeline.npy_to_multichannel_tiff(models, '800.00:1.0, 844.44:1.0, 888.89:1.0, 933.33:1.0, 977.78:1.0, 1022.22:1.0, 1066.67:1.0, 1111.11:1.0, 1155.56:1.0, 1200.00:1.0')

    # ------------测试NpyToReflectanceHdrConverter------------------
    # 测试保留
    # 反射板信息(要与场景中的虚拟反射板信息一致)
    BRFCurveCalculation.main(r'C:\Users\ZZZ\Desktop\22222222\result')
