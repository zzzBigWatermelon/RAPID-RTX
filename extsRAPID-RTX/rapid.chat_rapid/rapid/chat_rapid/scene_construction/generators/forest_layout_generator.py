from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from ..core.raycast import Raycast


class ForestLayoutGenerator:
    """
    Generate spatial layouts for forest vegetation.
    Responsibilities:
        1. Generate vegetation X/Y positions.
        2. Sample terrain height using Raycast.
        3. Return vegetation positions with terrain elevation.
    """

    def __init__(self, terrain_context: Dict[str, Any], tree_count: int, average_height: float,
                 species_ratio: Dict, seed: Optional[int] = None,):
        # 初始化参数
        self.terrain_context = terrain_context
        self.tree_count = int(tree_count)
        self.average_height = float(average_height)
        self.species_name_ratio = species_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Terrain information
        self.terrain_prim_path = self.terrain_context["prim_path"]
        self.size_x = float(self.terrain_context["size_x"])
        self.size_y = float(self.terrain_context["size_y"])

        # 高度采样入口
        self.raycast_sampler = Raycast()

    async def generate(self) -> List[Dict[str, Any]]:
        """
        Generate the complete forest spatial layout.
        Returns:
            List[Dict[str, Any]]: 每个树种一个字典的列表，结构如下：
                [
                    {
                        "species": "pine",          # 树种名称
                        "positions": np.ndarray,    # shape (M, 3)，每行 [x, y, z]
                        "normals": np.ndarray       # shape (M, 3)，每行 [nx, ny, nz]
                    },
                    ...
                ]
                其中 M 为该树种分配到的数量，所有 M 之和等于 self.tree_count
        """
        # 1. Generate X/Y positions
        xy_positions = self._generate_xy_positions(self.tree_count)

        # 2. Raycast terrain to obtain Z
        xyz_positions = await self._sample_terrain_xyz(xy_positions)

        # 3. Parse species distribution
        species_positions = self._assign_species_positions(xyz_positions)
        return species_positions

    def _generate_xy_positions(self, tree_count) -> np.ndarray:
        """
        Generate X/Y positions within the terrain boundary.
        This can later be replaced by:
            - Poisson disk sampling
            - clustered distribution
            - regular distribution
            - species-specific distribution
            - ecological spatial models
        """
        # 随机均匀分布,所有位置等概率生成
        x = self.rng.uniform(0.0, self.size_x, tree_count)
        y = self.rng.uniform(0.0, self.size_y, tree_count)
        return np.column_stack((x, y))

    async def _sample_terrain_xyz(self, xy_positions: np.ndarray,) -> np.ndarray:
        """
        Sample terrain height and surface normal using RTX raycast.
        Args:
            xy_positions:
                Nx2 array:
                [[x1, y1],
                 [x2, y2],
                 ...]
        Returns:
            Nx6 array:
            [[x, y, z, nx, ny, nz],
             [x, y, z, nx, ny, nz],
             ...]
        """
        xy_positions = np.asarray(xy_positions, dtype=np.float32)

        if xy_positions.ndim != 2 or xy_positions.shape[1] != 2:
            raise ValueError("xy_positions must have shape (N, 2).")

        xy_list = [(float(x), float(y)) for x, y in xy_positions]
        # 地形高度采样生成xyz坐标
        terrain_xyz = await self.raycast_sampler.sample_height(xy_list)
        return terrain_xyz.astype(np.float32)

    def _assign_species_positions(self, terrain_xyz: np.ndarray) -> List[Dict[str, Any]]:
        """
        根据物种比例将地形采样点分配给各个树种。
        Args:
            terrain_xyz: 地形采样结果数组,shape (N, 6)
                        每行包含 [x, y, z, normal_x, normal_y, normal_z]
                        其中 N = self.tree_count(总树木数量)
        Returns:
            List[Dict[str, Any]]: 每个树种一个字典的列表，结构如下：
                [
                    {
                        "species_name": "pine",          # 树种名称
                        "positions": np.ndarray,    # shape (M, 3)，每行 [x, y, z]
                        "normals": np.ndarray       # shape (M, 3)，每行 [nx, ny, nz]
                    },
                    ...
                ]
                其中 M 为该树种分配到的数量，所有 M 之和等于 self.tree_count
        """
        # 提取物种名称和比例
        species_names = list(self.species_name_ratio.keys())
        species_ratios = np.array(list(self.species_name_ratio.values()), dtype=np.float32)

        # 计算每个树种应得的数量（向下取整）
        counts = np.floor(species_ratios * self.tree_count).astype(int)
        remainder = self.tree_count - counts.sum()

        # 将余数分配给小数部分最大的几个物种
        if remainder > 0:
            fractions = species_ratios * self.tree_count - counts
            for i in np.argsort(-fractions)[:remainder]:
                counts[i] += 1

        # 随机打乱所有采样点索引
        indices = np.arange(self.tree_count)
        self.rng.shuffle(indices)

        # 按顺序为每个物种切分点
        layout = []
        start = 0
        for species, count in zip(species_names, counts):
            selected_indices = indices[start:start + count]
            selected_xyz = terrain_xyz[selected_indices]

            layout.append({
                "species_name": species,
                "positions": selected_xyz[:, :3].astype(np.float32),
                "normals": selected_xyz[:, 3:6].astype(np.float32),
            })

            start += count

        return layout
