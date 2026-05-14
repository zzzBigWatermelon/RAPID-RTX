# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import omni.ui as ui
from isaacsim.gui.components.ui_utils import get_style


class UIWidgetWrapper:
    """
    Base class for creating wrappers around any subclass of omni.ui.Widget in order to provide an easy interface
    for creating and managing specific types of widgets such as state buttons or file pickers.
    """

    def __init__(self, container_frame: ui.Frame):
        self._container_frame = container_frame

    @property
    def container_frame(self) -> ui.Frame:
        return self._container_frame

    @property
    def enabled(self) -> bool:
        return self.container_frame.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self.container_frame.enabled = value

    @property
    def visible(self) -> bool:
        return self.container_frame.visible

    @visible.setter
    def visible(self, value: bool):
        self.container_frame.visible = value

    def cleanup(self):
        """
        Perform any necessary cleanup
        """
        pass
