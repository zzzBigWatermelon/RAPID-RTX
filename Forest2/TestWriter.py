import io
import numpy as np
import omni.replicator.core as rep
from pxr import Usd
import json
import omni.kit.commands
import Semantics
from typing import Dict, Tuple

from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
from omni.syntheticdata._syntheticdata import acquire_syntheticdata_interface


# 这是一个RTX_Lidar专用的writer（编写器）模块
class RTX_LidarWriter(Writer):
    def __init__(
        self,
        output_dir,
        RTX_Lidar: bool = True,
        transformPoints: bool = True,
        semantic_types: bool = True,
        frame_padding: int = 8
    ):
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._sequence_id = 0
        self._frame_padding = frame_padding
        self.semantic_types = semantic_types
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()

        self.annotators = []

        # RTX_Lidar_Buffer
        if RTX_Lidar:
            # 添加LidarScanBuffer注册器
            annotator_RTX_Lidar = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            # 注册器初始参数
            annotator_RTX_Lidar.initialize(transformPoints=True,
                                           outputObjectId=True)
            self.annotators.append(annotator_RTX_Lidar)

    def write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        # Check for on_time triggers, data应该是继承的方法/变量，暂时不清楚sequence_id/trigger_outputs的含义
        # For each on_time trigger, prefix the output frame number with the trigger counts
        # 我是感觉底下的一段代码好像没啥用
        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        # 数据输出路径初始化
        for annotator in data.keys():
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0  # 这个参数应该只是用来判断是否有多种渲染输出，然后增加路径名称
            # multiple render_products
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]  # annotator_split[-1]应该是空字符，后边又写入了一次render_product_name
                render_product_path = f"{render_product_name}/"

            # 控制CreateRTXLidarScanBuffer注册器的数据输出
            if annotator.endswith("CreateRTXLidarScanBuffer"):
                if multi_render_prod:
                    render_product_path += "RTXLidarScanBuffer/"
                self.write_LidarScanBuffe(data, render_product_path, annotator)

        self._frame_id += 1

    def write_LidarScanBuffe(self, data: dict, render_product_path: str, annotator: str):
        # Lidar的xyz数据读取
        Lidar_data = data[annotator]['data']
        # 强度信息获取
        Lidar_data_intensity = data[annotator]['intensity']
        # objectId信息获取
        Lidar_data_objectId = data[annotator]['objectId']
        # self.write_semantic_types_json(Lidar_data_objectId, render_product_path)

        # 拼接数据的数组(需转置后才能拼接)
        Lidar_data_intensity = Lidar_data_intensity[:, np.newaxis]
        Lidar_data_objectId = Lidar_data_objectId[:, np.newaxis]
        Lidar_data = np.concatenate((Lidar_data, Lidar_data_intensity, Lidar_data_objectId), axis=1)

        # 输出路径、输出文件名self._frame_padding用于规定文件名后缀有几个0
        file_path = (
            f"{render_product_path}RTXLidarScanBuffer_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )

        # 二进制输出方式
        buf = io.BytesIO()
        np.save(buf, Lidar_data)
        self._backend.write_blob(file_path, buf.getvalue())

    # objectId转物体类型
    def write_semantic_types_json(self, Lidar_data_objectId, render_product_path):
        '''
        输入一个Lidar_data_objectId数组,
        objectId单独输出到一种npy文件, 再将对应的semantic信息输出到json文件
        '''

        # 获取Id信息，并剔除重复信息，获取唯一id
        objectId_unique_values = np.unique(Lidar_data_objectId)
        print('11111111111111111111111111111111111111111111111111111111111')
        print(objectId_unique_values)
        print('11111111111111111111111111111111111111111111111111111111111')
        # 获取标签信息
        segmentation_Labels = {}
        for objectId in objectId_unique_values:
            # 通过objectId查找prim路径
            primpath = acquire_syntheticdata_interface().get_uri_from_instance_segmentation_id(objectId)
            prim = self.stage.GetPrimAtPath(primpath)

            # 获取标签信息
            semantic_atrr_type, semantic_atrr_data = self.get_semantics(prim)
            segmentation_Labels[semantic_atrr_data] = int(objectId)

        # 将标签信息写入json文件
        Labels_file_path = (
            f"{render_product_path}segmentation_Labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        )
        buf = io.BytesIO()
        buf.write(json.dumps(segmentation_Labels).encode())
        self._backend.write_blob(Labels_file_path, buf.getvalue())

    def get_semantics(self, prim: Usd.Prim) -> Dict[str, Tuple[str, str]]:
        """输入一个USD的prim对象,返回他的分割标签(Semantics)信息.Returns semantics that are applied to a prim

        Args:
            prim (Usd.Prim): Prim to return semantics for

        Returns:
            Dict[str, Tuple[str,str]]: Dictionary containing the name of the applied semantic, and the type and data associated with that semantic.
        """
        result = {}
        for prop in prim.GetProperties():
            is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath())
            if is_semantic:
                name = prop.SplitName()[1]
                sem = Semantics.SemanticsAPI.Get(prim, name)

                typeAttr = sem.GetSemanticTypeAttr()
                dataAttr = sem.GetSemanticDataAttr()
                result[name] = (typeAttr.Get(), dataAttr.Get())
        # return result  原先返回字典对象
        return typeAttr.Get(), dataAttr.Get()


rep.WriterRegistry.register(RTX_LidarWriter)
