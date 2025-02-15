from pathlib import Path
import cv2 as cv
import os
import natsort as ns   # 用来给读取的图片文件排序
import numpy as np
# 读取json文件，复制器中间功能
import json
# 画图模块
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.api as sm
import shutil


class ReflectanceMeanGray:
    def __init__(self) -> None:
        # 输入输出路径
        Input_output_path = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'
        Input_output_abspath = os.path.abspath(Input_output_path)

        # 选择是否多次辐射定标（热点观测，太阳天顶角变化）
        multiple_observation = False
        if multiple_observation:
            # 将灰阶靶标的所有观测结果直接放入Input_output_path下
            target_dir_path_list = self.arrange_data(Input_output_abspath)
            for i in target_dir_path_list:  # 辐射定标
                self.main(i)
        else:
            self.main(Input_output_abspath)

    # 此方法将灰阶靶标的所有观测结果按照顺序分成文件夹，方便之后在每个文件夹中进行辐射定标操作
    def arrange_data(self, Input_output_path):
        target_dir_name_List = [str(i*2) for i in range(15, 46)]  # 生成目标文件夹的名称
        doc_name_List = ns.natsorted(os.listdir(Input_output_path))  # 读取源文件夹absPathGrayScale下所有文件名
        # 对所有文件进行分类
        bounding_box_list = doc_name_List[0:31]  # 包围盒边界数据文件.txt
        bounding_box_labels_list = doc_name_List[31:62]  # 包围盒标签数据文件.json
        print(bounding_box_labels_list)
        # png文件
        imageList = []
        for image in doc_name_List:
            if image.endswith('.png'):
                imageList.append(image)
        # dir_name_List和doc_name_List等长，所以采用列表下表将不同文件放入文件夹中,一共31个文件夹
        target_dir_path_list = []
        index = 0
        for target_dir_name in target_dir_name_List:
            # 生成目标文件夹
            target_dir_path = os.path.join(Input_output_path, target_dir_name)
            target_dir_path_list.append(target_dir_path)
            os.makedirs(target_dir_path, exist_ok=True)

            # 包围盒边界数据文件移动
            source = os.path.join(Input_output_path, bounding_box_list[index])
            target = os.path.join(target_dir_path, bounding_box_list[index])
            shutil.copy(source, target)  # 复制文件
            # 包围盒标签数据文件移动
            source = os.path.join(Input_output_path, bounding_box_labels_list[index])
            target = os.path.join(target_dir_path, bounding_box_labels_list[index])
            shutil.copy(source, target)
            # png数据文件移动
            source = os.path.join(Input_output_path, imageList[index])
            target = os.path.join(target_dir_path, imageList[index])
            shutil.copy(source, target)
            index += 1
        return target_dir_path_list

    # 辐射定标
    def main(self, Input_output_path):
        # 反射率-平均灰度值数据存储路径
        meanGrayDoc = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'/'MeanGray.txt'
        meanGrayDoc = os.path.join(Input_output_path, 'MeanGray.txt')
        # 线性拟合结果输出路径
        outcomeDoc = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'/'Outcome.txt'
        outcomeDoc = os.path.join(Input_output_path, 'Outcome.txt')
        # 源文件下所有文件名
        all_name_List = ns.natsorted(os.listdir(Input_output_path))
        # 提取图片名字列表
        imageList = []
        for image in all_name_List:
            if image.endswith('.png'):
                imageList.append(image)
        for docName in imageList:
            # 用于存储不同反射率灰阶靶标三个通道的灰度值，用于后续灰度值和反射率的线性回归
            R_meanGray = {}
            G_meanGray = {}
            B_meanGray = {}
            # ------------------------------提取到单个灰阶靶标的宽度-------------------------
            # 提取语义标签对应的id,转换成label与id的字典
            idFileName = str(docName).replace('rgb_', 'bounding_box_2d_tight_labels_')
            idFileName = idFileName.replace('.png', '.json')
            idFilePath = os.path.join(Input_output_path, idFileName)
            with open(idFilePath, 'r') as a:
                idList = json.load(a)
            labelDic = {}
            for id in idList.keys():
                labelDic[idList[id]['target']] = int(id)
            # 提取bounding_box文件
            boundingFileName = str(docName).replace('rgb_', 'bounding_box_2d_tight_')
            boundingFileName = boundingFileName.replace('.png', '.txt')
            boundingFilePath = os.path.join(Input_output_path, boundingFileName)
            with open(boundingFilePath, 'r') as f:
                boundingList = f.readlines()
            # -------------------------------分割灰阶靶标--------------------------------
            docPath = os.path.join(Input_output_path, docName)  # 得到文件的绝对路径
            src = cv.imread(docPath)
            src = cv.cvtColor(src, cv.COLOR_BGR2RGB)       # 转换成rgb色彩风格
            for id in boundingList:
                # 筛选语义id来区分每个灰阶靶标
                if float(id.split()[0]) == labelDic['gray_5']:
                    # 分割后的灰阶靶标命名
                    croppingImageName = str(docName).replace('.png', '')+"_gray_5"+'.png'
                    # 后面的MeanGray.txt文件第一列的内容
                    outputName = 'Gray-Scale='+str(docName).replace('.png', '')+'_5%'
                    # 设定R_meanGray等字典的key
                    meanGrayKey = '5%'
                elif float(id.split()[0]) == labelDic['gray_10']:
                    croppingImageName = str(docName).replace('.png', '')+"_gray_10"+'.png'
                    outputName = 'Gray-Scale='+str(docName).replace('.png', '')+'_10%'
                    meanGrayKey = '10%'
                elif float(id.split()[0]) == labelDic['gray_20']:
                    croppingImageName = str(docName).replace('.png', '')+"_gray_20"+'.png'
                    outputName = 'Gray-Scale='+str(docName).replace('.png', '')+'_20%'
                    meanGrayKey = '20%'
                elif float(id.split()[0]) == labelDic['gray_30']:
                    croppingImageName = str(docName).replace('.png', '')+"_gray_30"+'.png'
                    outputName = 'Gray-Scale='+str(docName).replace('.png', '')+'_30%'
                    meanGrayKey = '30%'
                elif float(id.split()[0]) == labelDic['gray_40']:
                    croppingImageName = str(docName).replace('.png', '')+"_gray_40"+'.png'
                    outputName = 'Gray-Scale='+str(docName).replace('.png', '')+'_40%'
                    meanGrayKey = '40%'
                elif float(id.split()[0]) == labelDic['gray_50']:
                    croppingImageName = str(docName).replace('.png', '')+"_gray_50"+'.png'
                    outputName = 'Gray-Scale='+str(docName).replace('.png', '')+'_50%'
                    meanGrayKey = '50%'
                else:
                    continue
                bounding_0 = id.split()
                x1 = int(float(bounding_0[1]))
                y1 = int(float(bounding_0[2]))
                x2 = int(float(bounding_0[3]))
                y2 = int(float(bounding_0[4]))
                hight = y2 - y1
                width = x2 - x1
                crop_img = src[y1:y1+hight, x1:x1+width]
                self.imageOutPutPath = os.path.join(Input_output_path, croppingImageName)
                cv.imwrite(self.imageOutPutPath, crop_img)
                # ----------------------------提取单个灰阶靶标的范围内的灰度值------------------------
                crop_imgInfo = crop_img.shape
                h = crop_imgInfo[0]
                w = crop_imgInfo[1]
                # 灰阶靶标的平行四边形对应的顶点坐标
                # Contour = np.array([[0, w], [0, w-boundingWidth+5], [h, 0], [h, boundingWidth-5]])
                # 将图像拆分成三个灰度图像
                (R, G, B) = cv.split(crop_img)
                # 用于存储每张图像RGB三个通道的灰度值，用于后续生成MeanGray文件
                meanGrayList = []
                # 新建一张测试图像，查看单个灰阶靶标的范围是否合格
                testImg = np.zeros((h, w, 1), src.dtype)
                testImgName = croppingImageName.replace('.png', 'linear.png')
                # 求每张图像的灰度总和
                for channel in (R, G, B):
                    Gray_sum = 0
                    grayScalePixel = 0
                    for i in range(5, h-5):
                        for j in range(5, w-5):
                            # if cv.pointPolygonTest(Contour, (i, j), False) == 1:
                            Gray = float('{:.2f}'.format(channel[i, j]))
                            # Gray = ((Gray+0.055)/1.055)**2.4  # 将灰度值从sRGB空间转换到线性(linear)空间计算
                            # Gray = float('{:.1f}'.format(Gray))
                            Gray_sum += Gray
                            grayScalePixel += 1
                            testImg[i, j] = Gray
                            # else:
                            # testImg[i, j] = 0
                    print(Gray_sum, grayScalePixel)
                    meanGray = '{0:.2f}'.format(float(Gray_sum/grayScalePixel))
                    meanGrayList.append(meanGray)
                # cv.imwrite(os.path.join(pathGrayScale, testImgName), testImg)
                # 存储每张影像不同通道的灰度值，用于后续灰度值和反射率的线性回归
                R_meanGray[meanGrayKey] = meanGrayList[0]
                G_meanGray[meanGrayKey] = meanGrayList[1]
                B_meanGray[meanGrayKey] = meanGrayList[2]
                # 保存反射率——平均灰度值的文件MeanGray.txt
                with open(meanGrayDoc, 'a') as outputTXT:
                    meanGrayoutput = (outputName, meanGrayList[0], meanGrayList[1], meanGrayList[2], '\r\n')
                    outputTXT.writelines(','.join(meanGrayoutput))
        # -----------------下面进行灰度值和反射率的线性回归，并生成一元线性回归图像------------
        predictList = []  # 用于存储三条拟合结果
        rsquaredList = []  # 同于存储三个拟合结果的标准差
        paramList = []  # 同于存储三个拟合结果的回归系数(截距和斜率)
        x = [0.05, 0.1, 0.2]  # 四个灰阶靶标的反射率值
        y_n_list = []   # 存储y_n，即将三个meanGray转化为按照key大小排序的列表
        for y in (R_meanGray, G_meanGray, B_meanGray):
            # 这里对meanGray内容按照key的大小排序，并且取出相应的value列表
            key_list = ns.natsorted(y)
            y_n = []
            for key in key_list:
                y_n.append(float(y[key]))
            y_n_list.append(y_n)
            # 线性拟合
            x_n = sm.add_constant(x)  # 若模型中有截距，必须有这一步
            model = sm.OLS(y_n, x_n).fit()  # 构建最小二乘模型并拟合
            predictList.append(model.predict())  # 储存结果的预测值
            rsq = model.rsquared   # 输出回归结果的R^2 （数）
            par = model.params  # 回归系数(截距和斜率)
            rsquaredList.append(rsq)
            paramList.append(par)
        # 将每个波段的拟合曲线数据输出
        band_list = ['R', 'G', 'B']
        with open(outcomeDoc, 'a') as outcome:
            for band in range(0, 3):
                slope = '{:.3f}'.format(paramList[band][1])  # 斜率
                intercept = '{:.3f}'.format(paramList[band][0])  # 截距
                rsquared = '{:.3f}'.format(rsquaredList[band])  # 平方差
                output = (band_list[band], slope, intercept, rsquared, '\r\n')
                outcome.writelines(','.join(output))
        # 生成一元线性回归图像
        style.use('ggplot')  # 加载'ggplot'风格
        plt.figure(figsize=(8, 6))  # 定义图的大小
        plt.xlabel("Reflectance")     # X轴标签
        plt.ylabel("DN")        # Y轴坐标标签
        plt.title("Radiometric calibration")      # 曲线图的标题'''
        # plt.xlim(0, 180)           # 设置x轴的范围
        # 画出真实值散点图
        plt.scatter(x, y_n_list[0], color="red", marker='p', label='R')
        plt.scatter(x, y_n_list[1], color="green", marker='*', label='G')
        plt.scatter(x, y_n_list[2], color="blue", marker='+', label='B')
        # 绘制函数,三条函数分别代表每张图像的R,G,B通道Reflectance-MeanGray拟合
        label_r = ['y=', '{0:.2f}'.format(paramList[0][1]), 'x', ' + ',  '{0:.2f}'.format(paramList[0][0]), '  ', 'R²=', '{0:.4f}'.format(rsquaredList[0])]
        label_g = ['y=', '{0:.2f}'.format(paramList[1][1]), 'x', ' + ',  '{0:.2f}'.format(paramList[1][0]), '  ', 'R²=', '{0:.4f}'.format(rsquaredList[1])]
        label_b = ['y=', '{0:.2f}'.format(paramList[2][1]), 'x', ' + ',  '{0:.2f}'.format(paramList[2][0]), '  ', 'R²=', '{0:.4f}'.format(rsquaredList[2])]
        plt.plot(x, predictList[0], color="red", label=''.join(label_r))
        plt.plot(x, predictList[1], color="green", label=''.join(label_g))
        plt.plot(x, predictList[2], color="blue", label=''.join(label_b))
        plt.legend(loc="best")  # 把标签加载到图中哪个位置, best 表示将标签加载到 python 认为最佳的位置
        # 存储图像
        outPath = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'/'Reflectance-MeanGray fitting curve.png'
        outPath = os.path.join(Input_output_path, 'Reflectance-MeanGray fitting curve.png')
        plt.savefig(outPath, dpi=700)


ReflectanceMeanGray()
