# core/raycast.py
import asyncio
import numpy as np
from typing import List, Tuple
import omni.kit.raycast.query


class Raycast:
    def __init__(self):
        self.results_3d = None
        self.completed_count = 0
        self.raycast_interface = None

    def _get_raycast(self):
        if self.raycast_interface is None:
            self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
        return self.raycast_interface

    async def sample_height(self, xy_list: List[Tuple[float, float]]) -> np.ndarray:
        '''
        异步批量射线查询地形高度和法线
        Args:
            xy_list: XY坐标列表,    如 [(x1,y1), (x2,y2), ...]
        Returns:
            np.ndarray: shape (num_rays, 6)，每行 [x, y, z, normal_x, normal_y, normal_z]
        '''
        num_rays = len(xy_list)
        self.results_3d = np.zeros((num_rays, 6), dtype=np.float32)
        self.completed_count = 0
        raycast = self._get_raycast()
        for i, (px, py) in enumerate(xy_list):
            ray = omni.kit.raycast.query.Ray((float(px), float(py), 1000000.0), (0.0, 0.0, -1.0))
            raycast.submit_raycast_query(ray, lambda r, res, idx=i, x=px, y=py: self._on_hit_sample_height(r, res, idx, x, y))
        while self.completed_count < num_rays:
            await asyncio.sleep(0.01)
        return self.results_3d

    def _on_hit_sample_height(self, ray, result, idx, x, y):
        '''
        射线命中回调，填充结果数组
        Args:
            ray: 射线对象
            result: 命中结果，包含 hit_position, normal, valid 等属性
            idx: 当前射线在 xy_list 中的索引
            x, y: 当前射线起点的 X Y 坐标
        Returns:
            None (直接修改 self.results_3d[idx])
        '''
        if result.valid:
            self.results_3d[idx] = [x, y, result.hit_position[2], result.normal[0], result.normal[1], result.normal[2]]
        else:
            self.results_3d[idx] = [x, y, 0.0, 0.0, 0.0, 1.0]
        self.completed_count += 1
