import omni
import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.commands
import omni.kit.app
from pathlib import Path
from pxr import Sdf, Usd
import os
# 读取json文件，复制器中间功能
import json
# 复制器核心模块
import omni.replicator.core as rep
# 调用其他自定义模块
from .generator import TreeGenerator, Delete, Load_LiDAR, Add_ScriptUAV, Scene_Construction, wind_Scene_Construction  # Load_backpack
from .BRF_displayWindow import BRF_plotExtension, WINDOW_TITLE
from .BRF_viewCamera import BRF_viewCamera, viewGrayScale
from .rtx_lidar import rtx_lidar
from .WorkWriter import WorkWriter
# 日志模块
'''import carb'''

# 激光雷达id标签文件夹
output_idToLabel_doc = Path(__file__).parent.parent/'result'/'pointcloud'/'idToLabel'
idToLabel_file = Path(__file__).parent.parent/'result'/'pointcloud'/'idToLabel'/'semantic_segmentation_labels_0000.json'


# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    print("[Forest2] some_public_function was called with x: ", x)
    return x ** x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class ForestExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id: str):
        print("[forest] forest startup")
        self._ext_id = ext_id
        self._stage: Usd.Stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()
        self.number_idToLabel = 0

        # 定义函数来输出一次复制器的结果，结果中包含语义id对应的标签
        # 复制器用于读取标签的功能
        def raplicator():
            camera_positions = [(0, 0, 15), (0, 0, 25), (0, 0, 35), (0, 0, 45), (0, 0, 55), (0, 0, 65)]
            camera = rep.create.camera(
                position=(0, 0, 15),
                look_at=(0, 0, 0),
                focus_distance=20,
                name='camera_GetLabel')
            render_product = rep.create.render_product(camera, (500, 500))
            with rep.trigger.on_frame(num_frames=6):
                with camera:
                    rep.modify.pose(position=rep.distribution.sequence(camera_positions), look_at=(0, 0, 0))
            # Initialize and attach writer
            writer = rep.WriterRegistry.get("WorkWriter")
            writer.initialize(
                output_dir=output_idToLabel_doc,
                rgb=True,
                semantic_segmentation=True,
                colorize_semantic_segmentation=False)
            writer.attach([render_product])
            rep.orchestrator.run()

        # -------------------------------------下面开始自定义UI窗口-----------------------------------------
        self._window = ui.Window("forest_generator", width=500, height=500)
        with self._window.frame:
            with ui.VStack():
                # -----------------------------------这是控制树木生成的UI-----------------------------------------
                with ui.HStack():
                    ui.Label("tree_Species: ")
                    self.tree_species_model = ui.IntField().model

                with ui.HStack():
                    ui.Label("tree_Width(m): ")
                    self.tree_width_model = ui.IntField().model
                    ui.Label("tree_Length(m): ")
                    self.tree_length_model = ui.IntField().model

                # 生成树木
                def on_click():
                    TreeGenerator(
                        TreeSpecies=self.tree_species_model.as_int,
                        width=self.tree_width_model.as_int,
                        length=self.tree_length_model.as_int,
                        )
                ui.Button("Tree", clicked_fn=lambda: on_click())

                # 删除树木
                def on_click_Delete():
                    Delete()
                ui.Button("Delete_Tree", clicked_fn=lambda: on_click_Delete())

                # RAMI场景重建
                def RAMI_Scene():
                    wind_Scene_Construction()
                ui.Button("RAMI_Scene", clicked_fn=lambda: RAMI_Scene())

            # ----------------------------------半圆运动相机，提取BRF--------------------------------
            # 先固定高度为15m，之后需要生成一系列的相机坐标，先设定相机是绕原点，从x轴正向旋转到x轴负向
                def BRDF_button():
                    BRF_viewCamera()  # 多角度观测

                def PlotBRF_button():
                    '''context = omni.usd.get_context()
                    current_file_path = context.get_stage_url()
                    context.open_stage(current_file_path)'''
                    # viewGrayScale()  # 观测一次灰阶靶标
                    rtx_lidar()
                    # BRF_plotExtension(WINDOW_TITLE)  # 将曲线图加载到一个新UI窗口
                with ui.HStack():
                    ui.Button("GrayScale", clicked_fn=lambda: PlotBRF_button())
                    ui.Button("BRF_Observation", clicked_fn=lambda: BRDF_button())

            # ----------------------------------加载UAV和读取id标签数据功能,lidar模型-------------------------------------
                # 运行复制器代码
                def raplicator_button(self):
                    # 清除之前的id标签文件
                    file_list = os.listdir(output_idToLabel_doc)
                    for file in file_list:
                        file_path = os.path.join(output_idToLabel_doc, file)
                        os.remove(file_path)
                    raplicator()

                # 加载UAV，并读取id标签文件
                def Load_UAV_button(self):
                    '''context = omni.usd.get_context()
                    current_file_path = context.get_stage_url()
                    context.open_stage(current_file_path)'''
                    # Add_ScriptUAV()
                    '''if os.path.exists(idToLabel_file):
                        with open(idToLabel_file, 'r') as f:
                            self.idToLabel = json.load(f)   # 返回字典对象
                    else:
                        self.idToLabel = 0'''
                with ui.HStack():
                    ui.Button("Semantic Segmentation", clicked_fn=lambda: raplicator_button(self))
                    ui.Button("Load_UAV", clicked_fn=lambda: Load_UAV_button(self))

    def on_shutdown(self):
        print("[Forest2] Forest2 shutdown")
        # Perform cleanup once the sample closes
        self._window = None
        self._editor_event_subscription_groundLidar = None
        self._editor_event_subscription_aerialLidar = None
