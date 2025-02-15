# import numpy as np
import cv2
from pathlib import Path
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
'''要根据使用方式不用,选择multiple_observation。并修改最后的VZA, 即观测次数和角度'''


class BRF:
    def __init__(self, inputPath=None, outputPath=None) -> None:
        self.inputPath = Path(__file__).parent.parent/'result'/'BRF'  # 图片输入的路径
        self.outputPat = outputPath  # 平均灰度值输出的路径
        if os.path.exists(Path(__file__).parent.parent/'result'/'BRF'/'metadata.txt'):
            os.remove(Path(__file__).parent.parent/'result'/'BRF'/'metadata.txt')

        self.multiple_observation_count(self.inputPath)

    # 处理以两种情况
    # 1、辐射定标后计算反射率，一个辐射定标文件，短时间的多次观测
    # 2、多个辐射定标文件，太阳天顶角变化很大的热点多次观测
    def multiple_observation_count(self, inputPath):
        multiple_observation = False
        # 将相对路径转换成绝对路径用于后续读取文件所在目录
        absPath = os.path.abspath(inputPath)
        nameList = os.listdir(absPath)

        # 提取出文件中的png图像
        imageList = []
        for image in nameList:
            if image.endswith('.png'):
                imageList.append(image)

        # 开始调用多次热点观测或正常BRF观测
        if multiple_observation:
            # 读取辐射定标结果
            fittingList_list = []  # 后面多次定标使用
            RadiometricCalibration_source_dir = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'
            RadiometricCalibration_dir_name = [str(i*2) for i in range(15, 46)]
            for index in RadiometricCalibration_dir_name:
                RadiometricCalibration_dir = os.path.join(RadiometricCalibration_source_dir, index)
                RadiometricCalibration_doc = os.path.join(RadiometricCalibration_dir, 'Outcome.txt')
                # 读取文件中的数据
                fittingList = []  # 列表三个元素放置了三个波段（波段，斜率，截距，平方差）包含数据的二维列表
                fittings = []  # 包含数据的维列表
                with open(RadiometricCalibration_doc, 'r') as s:
                    fitting = s.readlines()
                    for f in fitting:
                        if f != '\n':
                            fittings.append(f)
                for i in fittings:
                    fittingList.append(i.split(','))
                fittingList_list.append(fittingList)
            index = 0
            # 多次调用
            for image in imageList:
                self.meanGray(absPath, [image])
                self.BRF(fittingList_list[index])
                index += 1
        else:
            # 读取辐射定标结果
            RadiometricCalibration_doc = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'/'Outcome.txt'
            # 读取文件中的数据
            fittingList = []  # 列表三个元素放置了三个波段（波段，斜率，截距，平方差）包含数据的二维列表
            fittings = []  # 包含数据的维列表
            with open(RadiometricCalibration_doc, 'r') as s:
                fitting = s.readlines()
                for f in fitting:
                    if f != '\n':
                        fittings.append(f)
            for i in fittings:
                fittingList.append(i.split(','))
            # 单次调用
            self.meanGray(absPath, imageList)
            self.BRF(fittingList)

    # 求出每张图像指定立体角内的DN值（灰度值）
    def meanGray(self, Path, imageList):
        # 用于查看实际计算BRF用到的像素范围
        # testPath = Path(__file__).parent.parent/'result'/'BRF'/'test'  # 测试文件夹路径
        # testAbsPath = os.path.abspath(testPath)
        # 存储所有图像三个通道的DN值(灰度值)
        self.DN_List = []
        # BRF计算
        for docName in imageList:
            # 读取图像
            docPath = os.path.join(Path, docName)
            src = cv2.imread(docPath)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)       # 转换成rgb色彩风格
            # ----------------------------提取圆盘内固定立体角图像的DN值（灰度值）------------------------
            crop_imgInfo = src.shape
            h = crop_imgInfo[0]
            w = crop_imgInfo[1]
            # 将图像拆分成三个灰度图像
            (R, G, B) = cv2.split(src)
            # 用于存储单张图像RGB三个通道的灰度值，用于后续生成MeanGray文件
            meanGrayList = []
            # 新建一张测试图像，查看视场角的具体范围
            testImg = np.zeros((h, w, 1), src.dtype)
            # testName = str(docName).replace('.png', 'test1.png')
            test1Path = r'D:\Omniverse\Forest CODE2022.3.1\exts\Forest2\result\BRF\test\test.png'
            # 求每个通道的DN值总和
            for channel in (R, G, B):
                Gray_sum = 0
                Pixel = 0
                for i in range(h):
                    for j in range(w):
                        # 利用圆的半径判断一个点是否在视场角（圆）内,视场角17°，取10/8/6/4/2°做计算
                        # if math.sqrt((i-h/2)**2 + (j-w/2)**2) < w*((10/17)/2):
                            Gray = float('{:.2f}'.format(channel[i, j]))
                            Gray_sum += Gray
                            Pixel += 1
                            testImg[i, j] = Gray
                        # else:
                            # pass
                            # testImg[i, j] = 0
                meanGray = round(Gray_sum/Pixel, 4)
                meanGrayList.append(meanGray)
                print(Gray_sum, Pixel)
            '''在test文件夹不存在的情况下, 就不会输出test测试图像'''
            cv2.imwrite(test1Path, testImg)
            self.DN_List.append(meanGrayList)
        # print(R)
        # b = np.asarray(R)
        # BRFDoc = r'D:\Omniverse\Forest CODE2022.3.1\exts\Forest2\result\BRF\66.txt'
        # np.savetxt(BRFDoc, b)

    # -----------------------------------将每个对象的DN值转化为反射率---------------------------------------------
    def BRF(self, fittingList):
        '''# 读取反射率与灰度值的拟合结果
        outcomeDoc = Path(__file__).parent.parent/'result'/'BRF'/'Gray-Scale Targets'/'Outcome.txt'
        fittings = []  # 包含数据的维列表
        with open(outcomeDoc, 'r') as s:
            fitting = s.readlines()
            for f in fitting:
                if f != '\n':
                    fittings.append(f)
        for i in fittings:
            self.fittingList.append(i.split(','))'''
        # 提取每个波段的拟合结果的斜率和截距
        slope_R = float(fittingList[0][1])
        slope_G = float(fittingList[1][1])
        slope_B = float(fittingList[2][1])
        intercept_R = float(fittingList[0][2])
        intercept_G = float(fittingList[1][2])
        intercept_B = float(fittingList[2][2])
        # 保存单个波段所有不同角度反射率值,这个是按照图片顺序排序的
        Reflectance_R = []
        Reflectance_G = []
        Reflectance_B = []
        # 将每张图像的不同波段DN值转化成放射率
        for DN in self.DN_List:
            number = 0
            for DN_channel in DN:
                # R波段
                if number == 0:
                    Reflectance = (DN_channel-intercept_R)/slope_R
                    Reflectance_R.append(Reflectance)
                # G波段
                elif number == 1:
                    Reflectance = (DN_channel-intercept_G)/slope_G
                    Reflectance_G.append(Reflectance)
                # B波段
                else:
                    Reflectance = (DN_channel-intercept_B)/slope_B
                    Reflectance_B.append(Reflectance)
                number += 1
        # 数据输出
        BRFDoc = Path(__file__).parent.parent/'result'/'BRF'/'BRF.txt'
        # VZA = [a*10 for a in range(37)]  # 水平圆盘飞行
        VZA = [i for i in range(-60, 61)]  # 多太阳角度观测
        '''多太阳角度观测,多个辐射定标文件，一次只计算一张图片, VZA长度为1,但是多次循环, 以追加方式a写入数据'''
        # VZA = [1]  # 多太阳角度观测
        '''正常观测，一个辐射定标文件'''
        # VZA = [i*2 for i in range(15, 41)]  # 不同高度
        # VZA = [i for i in range(37)]
        # hotBRF = [a for a in range(0, 20)]
        # time = [i for i in range(1, 21)]  # 风立场模拟，时间序列定点观测
        index = 0
        with open(BRFDoc, 'w') as B:
            for vza in VZA:
                print(index)
                BRFdata_R = '{0:.4f}'.format(Reflectance_R[index])
                BRFdata_G = '{0:.4f}'.format(Reflectance_G[index])
                BRFdata_B = '{0:.4f}'.format(Reflectance_B[index])
                index += 1
                BRFdata = (str(vza), BRFdata_R, BRFdata_G, BRFdata_B, '\r\n')
                # BRFdata = (BRFdata_B, '\r\n')
                B.writelines(','.join(BRFdata))
                # B.writelines(BRFdata)


BRF()
