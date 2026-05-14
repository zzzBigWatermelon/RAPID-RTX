import os
import re
import cv2 as cv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


class RadiometricCalibration:

    def __init__(self, base_dir, skip_border: int = 1):
        """
        base_dir: 包含 Gray-Scale Targets 的主目录 (string or Path)
        skip_border: 裁剪后跳过边缘多少像素 (默认 5)
        """
        self.base_dir = str(base_dir)
        self.skip_border = skip_border

    def radiometric_calibration_pipeline(self,
                                         reflectance_values, reflectance_panel_keys_for_fit,
                                         reflectance_panel_Semantic_Type, reflectance_panel_Semantic_Data):
        """
        辐射定标的主流程：循环处理文件夹中的每一张图
        """
        # 查找npy结尾，HDR_radiometric_calibration开头的文件
        files = self.discover_files(self.base_dir, 'HDR_radiometric_calibration')

        for file_name in files:
            img_stem = os.path.splitext(file_name)[0]  # 获取文件名(不含后缀)，如 rgb_001
            print(f"Processing: {file_name}...")

            # 1. 提取单张图数据
            raw_data = self.process_single_radiometric_calibration_data(file_name, reflectance_panel_keys_for_fit, reflectance_panel_Semantic_Type, reflectance_panel_Semantic_Data)
            if not raw_data or not all(raw_data['R'].values()):
                print(f"  Skip {file_name}: Missing reflectance panels.")
                continue

            # 2. 聚合 (单图内聚合)
            mean_gray = self.aggregate_mean_gray(raw_data)

            # 3. 拟合
            # 检查此图中采集到的 key 是否完整，如果不全则无法拟合
            captured_keys = set(mean_gray['R'].keys())
            if not set(reflectance_panel_keys_for_fit).issubset(captured_keys):
                print(f"  Skip {file_name}: Not all gray targets found.")
                continue

            models = self.fit_reflectance(mean_gray, reflectance_values, reflectance_panel_keys_for_fit)

            # 4. 保存结果 (带上图片名前缀)
            outcome_path = os.path.join(self.base_dir, f"{img_stem}.txt")
            plot_path = os.path.join(self.base_dir, f"Fitting_{img_stem}.png")
            mean_gray_path = os.path.join(self.base_dir, f"MeanGray_{img_stem}.txt")

            self.save_regression_outcome(models, out_path=outcome_path)
            self.plot_calibration(mean_gray, models, reflectance_values, reflectance_panel_keys_for_fit, out_file=plot_path)
            self.save_mean_gray(mean_gray, out_path=mean_gray_path)
        return models

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
        return sorted(image_files, key=RadiometricCalibration.natural_key)

    def process_single_radiometric_calibration_data(self, data_name, reflectance_panel_keys_for_fit, reflectance_panel_Semantic_Type, reflectance_panel_Semantic_Data):
        """
        处理单个辐射定标文件，返回该图片的 DN 数据
        """
        processed_data = {'R': {}, 'G': {}, 'B': {}}
        data_path = os.path.join(self.base_dir, data_name)
        raw_RGBA_data = np.load(data_path)  # 默认的npy形状是（m,n,4）
        raw_data = raw_RGBA_data[:, :, :3].astype(np.float32)  # 取前三个通道的数据

        # 提取反射板的bbox和标签数据
        label_dict = self.load_label_dict(reflectance_panel_Semantic_Type)
        bbox_lines = self.load_bboxes()

        # 动态构建映射
        reflectance_map = {}
        for sem_data, gray_key in zip(reflectance_panel_Semantic_Data, reflectance_panel_keys_for_fit):
            sem_id = label_dict.get(sem_data)
            if sem_id is not None:
                reflectance_map[float(sem_id)] = gray_key

        for line in bbox_lines:
            parts = line.split()
            sem_id = float(parts[0])
            gray_key = reflectance_map.get(sem_id)
            if gray_key is None:
                continue

            x1, y1, x2, y2 = int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))
            try:
                band1, band2, band3 = self.compute_patch_mean_rgb(raw_data, (x1, y1, x2, y2))
            except ValueError:
                continue

            # 每张图每个反射率板通常只有一个框，所以这里直接存成列表
            processed_data['R'].setdefault(gray_key, []).append(band1)
            processed_data['G'].setdefault(gray_key, []).append(band2)
            processed_data['B'].setdefault(gray_key, []).append(band3)

        return processed_data

    def load_label_dict(self, reflectance_panel_Semantic_Type):
        '''
        读取WorkWriter输出的labels数据,json格式:{semanticId: {Semantic Type: Semantic Data},...}

        参数:
        image_name (str): 对应png图片的名字;

        返回:
        labels2id dict[str, str]: {Semantic Data: semanticId,...}
        '''
        json_path = os.path.join(self.base_dir, 'bounding_box_2d_tight_labels_0000.json')
        with open(json_path, 'r') as f:
            id_dict = json.load(f)
        return {v[reflectance_panel_Semantic_Type]: int(k) for k, v in id_dict.items()}

    def load_bboxes(self):
        """
        加载 bounding-box txt, txt的每一行对应('semanticId x_min y_min x_max y_max occlusionRatio')

        参数:
        image_name (str): 对应png图片的名字;

        返回:
        lines List[str]:['semanticId x_min y_min x_max y_max occlusionRatio',]
        """
        txt_path = os.path.join(self.base_dir, 'bounding_box_2d_tight_0000.txt')
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        return lines

    def compute_patch_mean_rgb(self, raw_data, bbox, skip_border=None):
        """
        给定 npy数据 (H, W, 3) 和 bbox (x1,y1,x2,y2),
        裁剪, 跳过边缘, 用 NumPy 向量化计算平均 R, G, B 值
        返回 (mean_R, mean_G, mean_B)
        """
        if skip_border is None:
            skip_border = self.skip_border
        x1, y1, x2, y2 = bbox
        crop = raw_data[y1:y2, x1:x2, :]
        sb = skip_border
        if crop.shape[0] <= 2*sb or crop.shape[1] <= 2*sb:
            raise ValueError(f"Crop too small for bbox {bbox}, skip_border {sb}")

        crop_inner = crop[sb:-sb, sb:-sb, :]
        mean_vals = crop_inner.reshape(-1, 3).mean(axis=0)
        return tuple(mean_vals.tolist())

    def process_all_images(self, reflectance_panel_keys_for_fit, reflectance_panel_Semantic_Type, reflectance_panel_Semantic_Data):
        """
        遍历所有 image, 读取标签 + bbox, 计算每个灰阶 patch 的 mean RGB,

        返回:
        dict data: { 'R': { reflectance_3: [vals ...],.. }, 'G':..., 'B':... }
        """
        data = {'R': {}, 'G': {}, 'B': {}}
        images = self.discover_files(self.base_dir)
        for img_name in images:
            img_path = os.path.join(self.base_dir, img_name)
            img_bgr = cv.imread(img_path)
            if img_bgr is None:
                print(f"Warning: failed to read {img_name}")
                continue
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

            label_dict = self.load_label_dict(img_name, reflectance_panel_Semantic_Type)
            bbox_lines = self.load_bboxes(img_name)

            reflectance_map = {
                label_dict.get(reflectance_panel_Semantic_Data[0]): reflectance_panel_keys_for_fit[0],
                label_dict.get(reflectance_panel_Semantic_Data[1]): reflectance_panel_keys_for_fit[1],
                label_dict.get(reflectance_panel_Semantic_Data[2]): reflectance_panel_keys_for_fit[2],
            }

            for line in bbox_lines:
                parts = line.split()
                sem_id = float(parts[0])
                gray_key = reflectance_map.get(sem_id)
                if gray_key is None:
                    continue
                x1, y1 = int(float(parts[1])), int(float(parts[2]))
                x2, y2 = int(float(parts[3])), int(float(parts[4]))
                try:
                    r, g, b = self.compute_patch_mean_rgb(img_rgb, (x1, y1, x2, y2))
                except ValueError as e:
                    print("Warning:", e)
                    continue
                data['R'].setdefault(gray_key, []).append(r)
                data['G'].setdefault(gray_key, []).append(g)
                data['B'].setdefault(gray_key, []).append(b)
        return data

    def aggregate_mean_gray(self, data):
        """
        对 process_all_images 的 output 做聚合 (mean over images/patches)
        返回 mean_gray: { 'R': {gray_key: mean_val}, 'G': ..., 'B': ... }
        """
        mean_gray = {}
        for band in ('R', 'G', 'B'):
            mean_gray[band] = {k: float(np.mean(v)) for k, v in data[band].items()}
        return mean_gray

    def save_mean_gray(self, mean_gray, out_path=None):
        """
        保存 mean_gray 到 MeanGray.txt
        """
        if out_path is None:
            out_path = os.path.join(self.base_dir, 'MeanGray.txt')
        with open(out_path, 'w') as f:
            for key in sorted(mean_gray['R'].keys(), key=lambda x: float(x.strip('%'))):
                r = mean_gray['R'][key]
                g = mean_gray['G'][key]
                b = mean_gray['B'][key]
                f.write(f"Gray-Scale={key},R={r:.2f},G={g:.2f},B={b:.2f}\n")

    def fit_reflectance(self, mean_gray, reflectance_values, gray_keys_for_fit):
        """
        对每个 band 做简单线性回归 (y = slope * x + intercept),
        使用 numpy.polyfit,不依赖 statsmodels。返回 models dict:
        { 'R': {'slope': ..., 'intercept': ..., 'r2': ..., 'y_pred': [...]}, ... }
        """
        models = {}
        x = np.array(reflectance_values, dtype=np.float32)
        for band in ('R', 'G', 'B'):
            y = np.array([mean_gray[band][k] for k in gray_keys_for_fit], dtype=np.float32)
            # 一元线性拟合（最低二乘）
            slope, intercept = np.polyfit(x, y, 1)
            # 计算预测值
            y_pred = slope * x + intercept
            # 计算 R²
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float('nan')
            models[band] = {
                'slope': slope,
                'intercept': intercept,
                'r2': r2,
                'y_pred': y_pred,
            }
        return models

    def save_regression_outcome(self, models, out_path=None):
        """
        将回归结果写 Outcome.txt
        """
        if out_path is None:
            out_path = os.path.join(self.base_dir, 'Outcome.txt')
        with open(out_path, 'w') as f:
            for band in ('R', 'G', 'B'):
                m = models[band]
                f.write(f"{band}, slope={m['slope']:.6f}, "
                        f"intercept={m['intercept']:.6f}, R2={m['r2']:.6f}\n")

    def plot_calibration(self, mean_gray, models, reflectance_values, gray_keys_for_fit, out_file=None):
        """
        画散点 + 拟合线 + 保存图像,
        """
        if out_file is None:
            out_file = os.path.join(self.base_dir, 'Reflectance_MeanGray_fitting_curve.png')
        style.use('ggplot')
        plt.figure(figsize=(8, 6))
        plt.xlabel("Reflectance")
        plt.ylabel("DN (mean gray)")
        plt.title("Radiometric calibration")

        x = np.array(reflectance_values, dtype=np.float32)

        for band, color, marker in zip(('R', 'G', 'B'),
                                    ('red', 'green', 'blue'),
                                    ('p', '*', '+')):
            y = np.array([mean_gray[band][k] for k in gray_keys_for_fit], dtype=np.float32)
            slope = models[band]['slope']
            intercept = models[band]['intercept']
            y_pred = models[band]['y_pred']
            r2 = models[band]['r2']

            plt.scatter(x, y, color=color, marker=marker, label=f"{band} data")
            plt.plot(x, y_pred, color=color,
                     label=f"{band}: y={slope:.2f}x + {intercept:.2f}, R²={r2:.4f}")

        plt.legend(loc="best")
        plt.savefig(out_file, dpi=700)
        plt.close()

    def load_models_from_outcome(self, outcome_path):
            """
            辅助函数：从 Outcome_rgb_xxxx.txt 中解析出 R, G, B 的 slope 和 intercept
            """
            models = {}
            if not os.path.exists(outcome_path):
                return None
            # 使用正则匹配：Band名称, slope, intercept
            # 匹配格式: R, slope=0.002241, intercept=-0.045053, R2=0.999971
            pattern = re.compile(r"([RGB]), slope=([\d\.\-]+), intercept=([\d\.\-]+)")

            with open(outcome_path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        band_name = match.group(1)
                        slope = float(match.group(2))
                        intercept = float(match.group(3))
                        models[band_name] = {'slope': slope, 'intercept': intercept}
            return models


# 测试保留
if __name__ == "__main__":
    # 反射板信息(要与场景中的虚拟反射板信息一致)
    REFLECTANCE_VALUES = [0.03, 0.05, 0.07]
    REFLECTANCE_KEYS = ['3%', '5%', '7%']
    reflectance_panel_Semantic_Type = 'reflectance_panel'
    reflectance_panel_Semantic_Data = ['reflectance_panel_3', 'reflectance_panel_5', 'reflectance_panel_7']

    # 处理流程
    base_dir = r'C:\Users\ZZZ\Desktop\22222222\intermediate_data'
    pipeline = RadiometricCalibration(base_dir=base_dir)
    models = pipeline.radiometric_calibration_pipeline(REFLECTANCE_VALUES, REFLECTANCE_KEYS, reflectance_panel_Semantic_Type, reflectance_panel_Semantic_Data)
    print(models)
