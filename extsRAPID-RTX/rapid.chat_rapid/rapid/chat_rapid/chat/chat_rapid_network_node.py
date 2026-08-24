from typing import List
from omni.ai.chat_usd.bundle.chat.chat_usd_network_node import ChatUSDNetworkNode


class ChatRAPIDNetworkNode(ChatUSDNetworkNode):
    """
    RAPID extension of Chat USD.

    Adds Scene Construction as a specialized expert.
    """

    default_node: str = "ChatRAPID_SupervisorNode"

    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "ChatRAPID_SceneConstruction",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metadata["description"] = (
            "RAPID AI extends Chat USD with scientific "
            "scene construction capabilities."
        )

        self.metadata["examples"] = [
            "Create a forest scene",
            "Generate terrain and populate it with trees",
            "Build a scientific scene for remote sensing",
        ]
