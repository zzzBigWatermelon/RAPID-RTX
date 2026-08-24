'''

'''
from typing import Optional
from ..generators.terrain_height_field_generator import TerrainHeightFieldGenerator
from ..builders.terrain_usd_builder import TerrainUSDBuilder


class GenerateTerrainTool:
    """
    RAPID-RTX Terrain Generation Tool.

    Generates a procedural terrain and creates the corresponding
    USD geometry in the current Omniverse stage.

    The generated terrain is intended to be used by downstream
    scene-construction tools, such as GenerateForestTool.

    Parameters:
        size_x:Terrain size along X in meters.
        size_y:Terrain size along Y in meters.
        slope:Overall terrain slope in degrees.
        roughness:Surface roughness from 0 to 1.
        resolution:Height field resolution.
        seed:Random seed for reproducible terrain generation.

    Return:
    """

    @staticmethod
    def run(terrain_type: str, size_x: float, size_y: float, slope: float = 0.0,
            roughness: float = 0.3, resolution: int = 128, seed: Optional[int] = None):
        # Generate terrain height field
        height_generator = TerrainHeightFieldGenerator(
            terrain_type=terrain_type,
            size_x=size_x,
            size_y=size_y,
            slope=slope,
            roughness=roughness,
            resolution=resolution,
            seed=seed)
        terrain_height_data = height_generator.generate()

        # 2. Build USD
        terrain_prim_path = "/World/Terrain"
        builder = TerrainUSDBuilder(prim_path=terrain_prim_path)
        builder.build(terrain_height_data, size_x, size_y)
        return {
            "status": "PASS",
            "message": "Terrain generated successfully.",
            "terrain": {
                "type": terrain_type,
                "size_x": size_x,
                "size_y": size_y,
                "slope": slope,
                "roughness": roughness,
                "resolution": resolution,
                "prim_path": terrain_prim_path}}
