from lc_agent import NetworkNode
from .scene_construction_modifier import SceneConstructionModifier


class SceneConstructionNetworkNode(NetworkNode):
    """
    Use this node to generate complete forest or terrain scenes based on natural language descriptions.
    It can create terrain, distribute trees, and output USD scenes ready for Omniverse.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_modifier(SceneConstructionModifier())

        # Set the default node to SceneConstructionNode (must be registered in factory)
        self.default_node = "SceneConstructionNode"

        # Rich metadata for Supervisor routing
        self.metadata["description"] = """
            Agent specialized in procedural forest and terrain scene construction.

            This agent handles requests involving:
            - Terrain generation
            - Forest generation
            - Tree distribution
            - Shrub and grass distribution
            - Vegetation placement
            - Combined terrain and vegetation scenes

            It is responsible for interpreting scene construction requests
            and coordinating the corresponding scene generation workflow.
            """

        self.metadata["examples"] = [
            "Create a terrain",
            "Create a forest scene",
            "Generate a pine forest on a gentle slope",
            "Create a terrain with scattered trees",
            "Generate a forest with trees and shrubs",
        ]
