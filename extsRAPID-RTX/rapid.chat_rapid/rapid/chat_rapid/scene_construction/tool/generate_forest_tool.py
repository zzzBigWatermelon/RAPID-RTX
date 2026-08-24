from typing import Dict, Any, Optional
from ..generators.forest_layout_generator import ForestLayoutGenerator
from ..distribution.vegetation_distributor import VegetationDistributor
from ..core.asset_registry import AssetRegistry
from ..core.material_manager import MaterialAssigner
import logging
logger = logging.getLogger(__name__)


class GenerateForestTool:
    """
    RAPID-RTX Forest Generation Tool.
    Pipeline:
        terrain_context
            ↓
        ForestLayoutGenerator
            ↓
        XYZ positions
            ↓
        AssetRegistry
            ↓
        VegetationDistributor
            ↓
        USD vegetation instances
    """
    def __init__(self):
        self.asset_registry = AssetRegistry()

    async def run(self, terrain_context: Dict[str, Any], tree_count: int, lai: float,
                  species_ratio: str, average_height: float, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate forest layout and distribute vegetation assets on existing terrain."""

        # 1. Parse species ratio，
        # 获取树种比例的字典dict，如{"American_Beech": 0.7, "Agave": 0.3}
        species_ratio_dict = self._parse_species_ratio(species_ratio)

        # 2. Generate forest layout
        # 生成空间分布
        layout_generator = ForestLayoutGenerator(
            terrain_context=terrain_context,
            tree_count=tree_count,
            average_height=average_height,
            species_ratio=species_ratio_dict,
            seed=seed,)
        layout = await layout_generator.generate()

        # 3. Check assets
        species_names = list(species_ratio_dict.keys())
        # 获取资产路径字典{'树种名':'usd模型磁盘路径'，...}
        species_assets_path = {}
        for species in species_names:
            assets_path = self.asset_registry.resolve_usd_path(species)
            species_assets_path[species] = assets_path

        # 获取对应树种的材质信息
        # material_RGB如{'树种名':{ "foliage_color":[0.1,0.1,0.1]，"trunk_color":[0.1,0.1,0.1]}，...}
        material_RGB = {}
        # material_mesh_rules是用于区分原型的mesh属于叶片还是枝干的作用
        # 如{'树种名':{"foliage": [], "trunk": []}....}
        material_mesh_rules = {}
        # 读取配置文件
        for species in species_names:
            material_RGB[species] = self.asset_registry.get_material_config(species)
            material_mesh_rules[species] = self.asset_registry.get_mesh_rules(species)

        # 4. Distribute vegetation
        distributor = VegetationDistributor(
            object_path=species_assets_path,
            distribution_data=layout,)
        distribution_result = distributor.run()
        # 从distribution结果中获取所有原型的舞台路径，汇总为 {species_name: prototype_path}
        prototype_paths_map = {item["species_name"]: item["prototype_path"] for item in distribution_result}

        # 5. Create material
        # 这个是给场景中的树给材质
        material_assigner = MaterialAssigner()
        await material_assigner.apply_species_materials(
            material_RGB=material_RGB,
            material_mesh_rules=material_mesh_rules,
            prototype_paths=prototype_paths_map)

        # 这个是给场景中的地形给材质
        # 准备数据
        terrain_material_RGB = {'terrain': {"foliage_color": [0.23, 0.12, 0.01],
                                            "trunk_color": [0.23, 0.12, 0.01]}}
        terrain_material_mesh_rules = {'terrain': {"foliage": ['terrain'], "trunk": ['terrain']}}
        terrain_prototype_paths = {'terrain': str(terrain_context["prim_path"])}  # 地形舞台路径
        # 开始执行
        await material_assigner.apply_species_materials(
            material_RGB=terrain_material_RGB,
            material_mesh_rules=terrain_material_mesh_rules,
            prototype_paths=terrain_prototype_paths)

        # 6. Return forest context
        return {
            "status": "PASS",
            "message": "Forest generated successfully.",
            "forest": {
                "tree_count": tree_count,
                "lai": lai,
                "average_height": average_height,
                "species_ratio": species_ratio,
                "terrain": {
                    "prim_path": terrain_context["prim_path"],
                    "size_x": terrain_context["size_x"],
                    "size_y": terrain_context["size_y"]},
                "layout_count": len(layout),
                "distribution": distribution_result,
            },
        }

    @staticmethod
    def _parse_species_ratio(species_ratio: str) -> Dict[str, float]:
        """
        Parse species ratio string into a dictionary mapping species name to its proportion.
        Args:
            species_ratio: A comma-separated string of species:ratio pairs.
                           Example: "American_Beech:0.7,Agave:0.3"

        Returns:
            Dict[str, float]: A dictionary where keys are species names and values are their ratios.
                              Example: {"American_Beech": 0.7, "Agave": 0.3}
        """
        result = {}
        for item in species_ratio.split(","):
            item = item.strip()
            if not item:
                continue

            parts = item.split(":")
            species = parts[0].strip()
            ratio = float(parts[1].strip())
            result[species] = ratio
        return result
