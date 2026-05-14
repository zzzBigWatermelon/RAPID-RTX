from omni.kit.window.property import PropertySchemeDelegate
from typing import List


class MyFilteredSchemeDelegate(PropertySchemeDelegate):
    def __init__(self, excluded_widgets=None):
        self._excluded_widgets = excluded_widgets or []

    def get_widgets(self, payload) -> List[str]:
        # 返回想显示的widget名称
        return [
            "transform",  
            "material_binding",
        ]

    def get_unwanted_widgets(self, payload) -> List[str]:
        # 明确排除不想要的widget
        return [
            "geometry",
            "kind",
            "semantics",
            "backdrop",  # 如果需要也排除
            'visual',
            'physx_custom_properties',
            "attribute",
            'isaac_array',
            'isaac_custom_data',
        ]