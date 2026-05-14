# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from pathlib import Path
from typing import Callable

import omni.appwindow
import omni.ext
import omni.kit
import omni.ui as ui
import omni.usd

ICON_FOLDER_PATH = Path(f"{omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)}/data")


class Dpad:
    def __init__(
        self,
        name="D-Pad Controller",
        clicked_fn_up: Callable = None,
        mouse_pressed_fn_up: Callable = None,
        mouse_released_fn_up: Callable = None,
        clicked_fn_down: Callable = None,
        mouse_pressed_fn_down: Callable = None,
        mouse_released_fn_down: Callable = None,
        clicked_fn_left: Callable = None,
        mouse_pressed_fn_left: Callable = None,
        mouse_released_fn_left: Callable = None,
        clicked_fn_right: Callable = None,
        mouse_pressed_fn_right: Callable = None,
        mouse_released_fn_right: Callable = None,
        clicked_fn_cw: Callable = None,
        mouse_pressed_fn_cw: Callable = None,
        mouse_released_fn_cw: Callable = None,
        clicked_fn_ccw: Callable = None,
        mouse_pressed_fn_ccw: Callable = None,
        mouse_released_fn_ccw: Callable = None,
        width=210,
        height=210,
    ):
        """D-Pad Controller Class

        Args:
            name (str, optional): Window Title. Defaults to "D-Pad Controller".
            clicked_fn_up (Callable, optional): Sets the function that will be called when when the button is activated
            (i.e., pressed down then released while the mouse cursor is inside the button). Defaults to None.
            mouse_pressed_fn_up (Callable, optional): Sets the function that will be called when the user presses the
            mouse button inside the widget. The function should be like this:
            void onMousePressed(float x, float y, int32_t button, carb::input::KeyboardModifierFlags modifier)
            Where 'button' is the number of the mouse button pressed. 'modifier' is the flag for the keyboard modifier key.
            Defaults to None.
            mouse_released_fn_up (Callable, optional): Sets the function that will be called when the user releases the
            mouse button if this button was pressed inside the widget. The function should be like this:
            void onMouseReleased(float x, float y, int32_t button, carb::input::KeyboardModifierFlags modifier)
            Defaults to None.
            clicked_fn_down (Callable, optional): On Click Function. Defaults to None.
            mouse_pressed_fn_down (Callable, optional): On Mouse Pressed Function. Defaults to None.
            mouse_released_fn_down (Callable, optional): On Mouse Released Function. Defaults to None.
            clicked_fn_left (Callable, optional): On Click Function. Defaults to None.
            mouse_pressed_fn_left (Callable, optional): On Mouse Pressed Function. Defaults to None.
            mouse_released_fn_left (Callable, optional): On Mouse Released Function. Defaults to None.
            clicked_fn_right (Callable, optional): On Click Function. Defaults to None.
            mouse_pressed_fn_right (Callable, optional): On Mouse Pressed Function. Defaults to None.
            mouse_released_fn_right (Callable, optional): On Mouse Released Function. Defaults to None.
            clicked_fn_cw (Callable, optional): On Click Function. Defaults to None.
            mouse_pressed_fn_cw (Callable, optional): On Mouse Pressed Function. Defaults to None.
            mouse_released_fn_cw (Callable, optional): On Mouse Released Function. Defaults to None.
            clicked_fn_ccw (Callable, optional): On Click Function. Defaults to None.
            mouse_pressed_fn_ccw (Callable, optional): On Mouse Pressed Function. Defaults to None.
            mouse_released_fn_ccw (Callable, optional): On Mouse Released Function. Defaults to None.
            width (int, optional): Window Width. Defaults to 210.
            height (int, optional): Window Height. Defaults to 210.
        """
        self.name = name

        self.clicked_fn_up = clicked_fn_up
        self.mouse_pressed_fn_up = mouse_pressed_fn_up
        self.mouse_released_fn_up = mouse_released_fn_up

        self.clicked_fn_down = clicked_fn_down
        self.mouse_pressed_fn_down = mouse_pressed_fn_down
        self.mouse_released_fn_down = mouse_released_fn_down

        self.clicked_fn_left = clicked_fn_left
        self.mouse_pressed_fn_left = mouse_pressed_fn_left
        self.mouse_released_fn_left = mouse_released_fn_left

        self.clicked_fn_right = clicked_fn_right
        self.mouse_pressed_fn_right = mouse_pressed_fn_right
        self.mouse_released_fn_right = mouse_released_fn_right

        self.clicked_fn_cw = clicked_fn_cw
        self.mouse_pressed_fn_cw = mouse_pressed_fn_cw
        self.mouse_released_fn_cw = mouse_released_fn_cw

        self.clicked_fn_ccw = clicked_fn_ccw
        self.mouse_pressed_fn_ccw = mouse_pressed_fn_ccw
        self.mouse_released_fn_ccw = mouse_released_fn_ccw

        self.width = width
        self.height = height
        self._build_ui()
        return

    def _build_ui(self):
        self._window = ui.Window(
            title=self.name,
            width=self.width,
            height=self.height,
            visible=True,
            dockPreference=ui.DockPreference.LEFT,
            auto_resize=True,
        )
        with self._window.frame:
            padding = 6
            h_spacing = self.width / 3.0 - padding
            v_spacing = h_spacing
            with ui.VStack(spacing=0, height=self.height, width=self.width):
                with ui.HStack():
                    if self.clicked_fn_ccw or self.mouse_pressed_fn_ccw or self.mouse_released_fn_ccw:
                        ui.Button(
                            clicked_fn=self.clicked_fn_ccw,
                            mouse_pressed_fn=self.mouse_pressed_fn_ccw,
                            mouse_released_fn=self.mouse_released_fn_ccw,
                            image_width=h_spacing,
                            image_height=h_spacing,
                            image_url=f"{ICON_FOLDER_PATH}/arrow_counterclockwise.svg",
                            alignment=ui.Alignment.LEFT_CENTER,
                        )
                    else:
                        ui.Spacer()
                    ui.Button(
                        clicked_fn=self.clicked_fn_up,
                        mouse_pressed_fn=self.mouse_pressed_fn_up,
                        mouse_released_fn=self.mouse_released_fn_up,
                        image_width=h_spacing,
                        image_height=h_spacing,
                        image_url=f"{ICON_FOLDER_PATH}/arrow_up.svg",
                        alignment=ui.Alignment.CENTER,
                    )
                    if self.clicked_fn_cw or self.mouse_pressed_fn_cw or self.mouse_released_fn_cw:
                        ui.Button(
                            clicked_fn=self.clicked_fn_cw,
                            mouse_pressed_fn=self.mouse_pressed_fn_cw,
                            mouse_released_fn=self.mouse_released_fn_cw,
                            image_width=h_spacing,
                            image_height=h_spacing,
                            image_url=f"{ICON_FOLDER_PATH}/arrow_clockwise.svg",
                            alignment=ui.Alignment.RIGHT_CENTER,
                        )
                    else:
                        ui.Spacer()
                with ui.HStack(spacing=0):
                    ui.Button(
                        clicked_fn=self.clicked_fn_left,
                        mouse_pressed_fn=self.mouse_pressed_fn_left,
                        mouse_released_fn=self.mouse_released_fn_left,
                        image_width=h_spacing,
                        image_height=h_spacing,
                        image_url=f"{ICON_FOLDER_PATH}/arrow_left.svg",
                        alignment=ui.Alignment.LEFT_CENTER,
                    )
                    ui.Spacer()
                    ui.Button(
                        clicked_fn=self.clicked_fn_right,
                        mouse_pressed_fn=self.mouse_pressed_fn_right,
                        mouse_released_fn=self.mouse_released_fn_right,
                        image_width=h_spacing,
                        image_height=h_spacing,
                        image_url=f"{ICON_FOLDER_PATH}/arrow_right.svg",
                        alignment=ui.Alignment.RIGHT_CENTER,
                    )
                with ui.HStack():
                    ui.Spacer()
                    ui.Button(
                        clicked_fn=self.clicked_fn_down,
                        mouse_pressed_fn=self.mouse_pressed_fn_down,
                        mouse_released_fn=self.mouse_released_fn_down,
                        image_width=h_spacing,
                        image_height=h_spacing,
                        image_url=f"{ICON_FOLDER_PATH}/arrow_down.svg",
                        alignment=ui.Alignment.CENTER_BOTTOM,
                    )
                    ui.Spacer()
        return

    def shutdown(self):
        self._window = None
        self.name = None
        self.width = None
        self.height = None
        return
