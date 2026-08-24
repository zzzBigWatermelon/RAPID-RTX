import json
from pathlib import Path
import carb.tokens


class AssetRegistry:
    """
    Registry for RAPID-RTX vegetation assets.
    Provides access to vegetation assets defined in asset_registry.json.
    """

    def __init__(self, registry_path=None):
        # 如果未指定路径，使用基于 ${kit} 的默认路径
        if registry_path is None:
            # 获取 tokens 接口，获取程序的根目录位置
            tokens = carb.tokens.get_tokens_interface()
            kit_path = tokens.resolve("${kit}")
            kit_root = Path(kit_path).parent  # 根据实际目录结构调整
            registry_path = kit_root / "assets_rapid-rtx" / "models" / "vegetation" / "asset_registry.json"

        # 读取json文件
        self.registry_path = Path(registry_path)
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.asset_root = self.registry_path.parent

    def get_species_descriptions(self):
        """
        Return concise metadata for LLM prompt construction.
        用于给AI模型返回资产库中的所有植被信息,让他添加到系统提示词中

        Return
        result (dic):{
                "species": species,
                "type": asset.get("type", []),
                "description": asset.get("description", "")}
        """
        result = []

        for species, asset in self.data.items():
            result.append({
                "species": species,
                "type": asset.get("type", []),
                "description": asset.get("description", "")
            })
        return result

    def get_assets(self, species):
        """
        Return all assets for a species.
        返回资产库中指定的植被的所有信息
        Parameter:
        """
        asset = self.data.get(species)
        if asset is None:
            raise ValueError(f"Unknown vegetation species: {species}")
        return asset

    def validate_species(self, species_list):
        """
        验证所有请求的物种是否都存在于植被资产登记册中。
        Validate whether all requested species exist in the vegetation asset registry.
        Parameter:
        species_list (list):所有请求的物种的字符串列表

        Return:
        result (dic):valid和invalid包含的树种列表
                {"valid": valid,
                "valid": invalid}

        """
        valid = []
        invalid = []

        for species in species_list:
            if self.has_species(species):
                valid.append(species)
            else:
                invalid.append(species)

        return {"valid": valid,
                "invalid": invalid}

    def resolve_usd_path(self, species):
        """
            Return the USD asset path for a species.
            返回指定树种3D模型的资产路径
        """
        asset = self.get_assets(species)
        usd_path = asset.get("model_path")

        if not usd_path:
            raise ValueError(
                f"No USD path defined for species: {species}")
        return str((self.asset_root / usd_path).resolve())

    def get_material_config(self, species):
        """
        Return material configuration for a species.
        返回指定树种的材质配置。
        """
        asset = self.get_assets(species)
        material = asset.get("material", {})
        return {
            "foliage_color": material.get("foliage_color"),
            "trunk_color": material.get("trunk_color"),
        }

    def get_mesh_rules(self, species):
        """
        Return mesh matching rules for a species.
        返回指定树种的 Mesh 匹配规则。
        """
        asset = self.get_assets(species)
        return asset.get("mesh_rules", {"foliage": [], "trunk": []})
