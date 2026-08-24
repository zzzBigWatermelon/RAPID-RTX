import omni.ext
from lc_agent import get_node_factory
from .chat.chat_rapid_network_node import ChatRAPIDNetworkNode
from .chat.chat_rapid_supervisor import ChatRAPIDSupervisorNode
from .scene_construction.scene_construction_network_node import SceneConstructionNetworkNode
from .scene_construction.scene_construction_node import SceneConstructionNode


class ChatRAPIDExtension(omni.ext.IExt):

    def on_startup(self, ext_id):

        self._node_factory = get_node_factory()

        # Register RAPID Supervisor
        self._node_factory.register(
            ChatRAPIDSupervisorNode,
            name="ChatRAPID_SupervisorNode",
            hidden=True,
        )
        # Register Scene Construction Runnable Node
        self._node_factory.register(
            SceneConstructionNode,
            name="SceneConstructionNode",
            hidden=True,
        )
        # Register Scene Construction Expert
        self._node_factory.register(
            SceneConstructionNetworkNode,
            name="ChatRAPID_SceneConstruction",
            hidden=True,
        )

        # Register RAPID Chat USD
        self._node_factory.register(
            ChatRAPIDNetworkNode,
            name="Chat RAPID",
            multishot=True,
        )

    def on_shutdown(self):

        self._node_factory.unregister("Chat RAPID")
        self._node_factory.unregister("ChatRAPID_SceneConstruction")
