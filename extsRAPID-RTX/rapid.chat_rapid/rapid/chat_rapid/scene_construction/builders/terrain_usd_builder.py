from typing import Dict, Any, Optional
import numpy as np
from pxr import UsdGeom, Gf
import omni.usd


class TerrainUSDBuilder:
    """
    Build a USD terrain mesh from generated terrain data.

    Responsibilities:
        TerrainData -> USD Mesh

    This class does not:
        - parse AI commands
        - generate terrain heights
        - generate tree positions
        - perform forest distribution
    """

    def __init__(self, prim_path: str = "/World/Terrain"):
        self.stage = omni.usd.get_context().get_stage()
        self.prim_path = prim_path

    def build(self, heights: np.ndarray, size_x: float, size_y: float):
        """
        Build terrain mesh in the current USD stage.
        Expected terrain_data:

        """
        terrain_data = TerrainUSDBuilder.heightfield_to_mesh_data(heights, size_x, size_y)

        # 构建 USD Mesh
        points = [Gf.Vec3f(v[0], v[1], v[2]) for v in terrain_data["vertices"]]
        face_vertex_counts = [3] * len(terrain_data["faces"])  # 全部是三角形
        face_vertex_indices = []
        for face in terrain_data["faces"]:
            face_vertex_indices.extend(face)

        # Create USD Mesh
        mesh = UsdGeom.Mesh.Define(self.stage, self.prim_path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)


    @staticmethod
    def heightfield_to_mesh_data(heights: np.ndarray, size_x: float, size_y: float) -> dict:
        """
        将高度场矩阵转换为顶点、面片列表。
        """
        resolution = heights.shape[0]
        vertices = []
        faces = []

        # 1. 生成顶点
        for iy in range(resolution):
            for ix in range(resolution):
                x = (ix / (resolution - 1)) * size_x
                y = (iy / (resolution - 1)) * size_y
                z = heights[iy, ix]
                vertices.append((x, y, z))  # USD 坐标: (X, Z, Y)

        # 2. 生成三角形面片
        for iy in range(resolution - 1):
            for ix in range(resolution - 1):
                idx00 = iy * resolution + ix
                idx10 = iy * resolution + (ix + 1)
                idx01 = (iy + 1) * resolution + ix
                idx11 = (iy + 1) * resolution + (ix + 1)
                faces.append([idx00, idx10, idx01])
                faces.append([idx10, idx11, idx01])

        return {"vertices": vertices, "faces": faces}
