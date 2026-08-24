from pathlib import Path

from lc_agent import RunnableNode, RunnableSystemAppend
from omni.ai.chat_usd.bundle.chat.chat_usd_network_node import (
    ChatUSDSupervisorNode,
)

SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


identity = read_md_file(
    f"{SYSTEM_PATH}/chat_rapid_supervisor_identity.md"
)


class ChatRAPIDSupervisorNode(ChatUSDSupervisorNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 替换 / 追加 RAPID 自己的 Supervisor System Message
        self.inputs.append(
            RunnableSystemAppend(
                system_message=identity
            )
        )

