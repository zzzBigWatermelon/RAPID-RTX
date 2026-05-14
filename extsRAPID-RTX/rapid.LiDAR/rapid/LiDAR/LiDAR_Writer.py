import io
import numpy as np
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Gf
from pxr import Usd
import omni.kit.commands
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch


# 这是一个RTX_Lidar专用的writer（编写器）模块
class RTX_LiDARWriter(Writer):
    def __init__(
        self,
        output_dir,
        lidar_path,
        RTX_LiDAR: bool = True,
        frame_padding: int = 8
    ):
        self._output_dir = output_dir
        self._lidar_path = lidar_path  # 存储Lidar路径
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._frame_padding = frame_padding

        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        self.annotators = []
        # RTX_Lidar_Buffer
        if RTX_LiDAR:
            # 添加LidarScanBuffer注册器
            annotator_RTX_Lidar = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
            # 注册器初始参数
            annotator_RTX_Lidar.initialize(transformPoints=False)  # 明确输出Local坐标
            self.annotators.append(annotator_RTX_Lidar)

    def write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        render_product_path = ""
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
            if annotator.endswith("PointCloudNoAccumulator"):
                if multi_render_prod:
                    render_product_path += "PointCloudNoAccumulator/"
                # 提取当前的位姿矩阵 (World Transform)
                world_matrix = self._get_world_transform()

                # 调用增强后的写入函数
                self.write_LidarData(data, render_product_path, annotator, world_matrix)

        self._frame_id += 1

    def _get_world_transform(self):
        """获取 Lidar 的世界变换矩阵，确保产出标准的 4x4 行主序矩阵"""
        prim = self.stage.GetPrimAtPath(self._lidar_path)

        # 获取当前时间码对象
        from pxr import Usd
        current_time_code = Usd.TimeCode(self.timeline.get_current_time() * self.timeline.get_time_codes_per_seconds())

        xformable = UsdGeom.Xformable(prim)
        # 传入 TimeCode 对象
        world_transform_gf = xformable.ComputeLocalToWorldTransform(current_time_code)

        return np.array(world_transform_gf)

    def write_LidarData(self, data: dict, render_product_path: str, annotator: str, world_matrix):
        # Lidar的xyz数据读取
        Lidar_data = data[annotator]['data']

        # 输出路径、输出文件名self._frame_padding用于规定文件名后缀有几个0
        pcd_file_path = (
            f"{render_product_path}RTXLidarScanBuffer_{self._frame_id:0{self._frame_padding}}.npy"
        )
        # 二进制输出点云文件
        pcd_buf = io.BytesIO()
        np.save(pcd_buf, Lidar_data)
        self._backend.write_blob(pcd_file_path, pcd_buf.getvalue())

        # 二进制输出世界坐标矩阵文件
        matrix_file_path = (
            f"{render_product_path}WorldMatrix_{self._frame_id:0{self._frame_padding}}.npy"
        )
        matrix_buf = io.BytesIO()
        np.save(matrix_buf, world_matrix)
        self._backend.write_blob(matrix_file_path, matrix_buf.getvalue())


rep.WriterRegistry.register(RTX_LiDARWriter)
