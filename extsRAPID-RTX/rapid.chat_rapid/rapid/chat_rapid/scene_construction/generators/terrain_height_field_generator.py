from typing import Optional
import numpy as np


class TerrainHeightFieldGenerator:
    """
    Generate procedural terrain height fields.
    The first version uses:base slope+multi-scale terrain noise
    to create a simple procedural terrain.
    """

    def __init__(self, terrain_type: str, size_x: float, size_y: float, slope: float = 0.0, roughness: float = 0.3,
                 resolution: int = 128, seed: Optional[int] = None):

        # 初始化参数
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.slope = float(slope)
        self.roughness = float(roughness)
        self.resolution = int(resolution)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self):
        """
        Generate a Terrain object.
        """
        # 1. Base slope
        x = np.linspace(0.0, self.size_x, self.resolution)
        y = np.linspace(0.0, self.size_y, self.resolution)
        xx, yy = np.meshgrid(x, y)

        # 1. Base slope
        slope_rad = np.deg2rad(self.slope)
        base_height = (xx * np.tan(slope_rad))

        # 2. Generate terrain noise
        noise = self._generate_noise(self.resolution, self.resolution)

        # 3. Scale noise
        # Terrain size controls the natural scale of the variation.
        characteristic_length = min(self.size_x, self.size_y)
        noise_amplitude = (characteristic_length * 0.05 * self.roughness)
        terrain_height = (base_height + noise * noise_amplitude)

        # 4. Normalize minimum elevation
        terrain_height -= terrain_height.min()

        # 5. Create Terrain object
        return terrain_height

    def _generate_noise(self, height: int, width: int) -> np.ndarray:
        """
        Generate multi-scale smooth random noise.
        This is intentionally lightweight and does not require
        an external noise library.
        """
        noise = np.zeros((height, width), dtype=np.float32,)
        # Multiple spatial frequencies
        octaves = [(4, 1.0), (8, 0.5), (16, 0.25), (32, 0.125)]
        total_weight = 0.0

        for grid_size, weight in octaves:
            small_h = max(2, grid_size)
            small_w = max(2, grid_size)
            small_noise = self.rng.normal(0.0, 1.0, size=(small_h, small_w))
            resized = self._resize_noise(small_noise, height, width)
            noise += resized * weight
            total_weight += weight

        if total_weight > 0:
            noise /= total_weight

        # Normalize to approximately [-1, 1]
        min_value = noise.min()
        max_value = noise.max()

        if max_value - min_value > 1e-8:
            noise = (2.0 * (noise - min_value) / (max_value - min_value) - 1.0)
        return noise

    @staticmethod
    def _resize_noise(
        source: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Resize a low-resolution noise field using
        bilinear interpolation.
        """
        source_h, source_w = source.shape

        y = np.linspace(0, source_h - 1, target_h)
        x = np.linspace(0, source_w - 1, target_w)

        # Interpolate along X
        temp = np.empty(
            (source_h, target_w),
            dtype=np.float32,
        )

        source_x = np.arange(
            source_w
        )

        for i in range(source_h):
            temp[i] = np.interp(
                x,
                source_x,
                source[i],
            )

        # Interpolate along Y
        result = np.empty(
            (target_h, target_w),
            dtype=np.float32,
        )

        source_y = np.arange(
            source_h
        )

        for j in range(target_w):
            result[:, j] = np.interp(
                y,
                source_y,
                temp[:, j],
            )

        return result