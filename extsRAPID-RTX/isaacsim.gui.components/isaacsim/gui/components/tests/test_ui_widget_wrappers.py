# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# This import is included for visualization of UI elements as demonstrated in testXYPlotWrapper
import asyncio

import numpy as np
import omni.kit.test
import omni.kit.ui_test as ui_test
import omni.timeline
import omni.ui as ui
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, update_stage_async
from isaacsim.gui.components.element_wrappers import (
    Button,
    CheckBox,
    CollapsableFrame,
    ColorPicker,
    DropDown,
    FloatField,
    Frame,
    IntField,
    ScrollingFrame,
    ScrollingWindow,
    StateButton,
    StringField,
    TextBlock,
    XYPlot,
)
from isaacsim.storage.native import get_assets_root_path


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestUI(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        pass

    # After running each test
    async def tearDown(self):
        await update_stage_async()
        pass

    async def _create_window(self, title, width, height):
        window = ScrollingWindow(
            title=title,
            width=width,
            height=height,
            visible=True,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        await update_stage_async()
        return window

    async def testButtonWrapper(self):
        # Width and height chosen such that Button is visible in UI Window
        window_title = "UI_Widget_Wrapper_Test_Window_Button_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.btn_counter = 0

        def on_click_fn():
            self.btn_counter += 1

        with window.frame:
            btn = Button("Button", "BUTTON", on_click_fn=on_click_fn)

        button = ui_test.find(f"{window_title}//Frame/Frame[0]/HStack[0]/Button[0]")
        await button.click()

        self.assertTrue(self.btn_counter == 1, "Button is not working as expected")

        btn.trigger_click()

        self.assertTrue(self.btn_counter == 2, "Button is not working as expected")

        window.destroy()

    async def testCheckBoxWrapper(self):
        # There is buggy behavior of the ui_test.find(...).click() function
        # where it clicks the wrong place on the screen for checkboxes
        # The width and height chosen here happen to get it to click the right part of the screen

        window_title = "UI_Widget_Wrapper_Test_Window_CheckBox_Test"
        width = 200
        height = 200
        window = await self._create_window(window_title, width, height)

        self.cb_callbacks = []

        def on_click_fn(value):
            self.cb_callbacks.append(value)

        with window.frame:
            with CollapsableFrame("", collapsed=False):
                self.cb_wrapper = CheckBox("Checkbox", True, on_click_fn=on_click_fn)

        cb = ui_test.find(
            f"{window_title}//Frame/CollapsableFrame[0]/Frame[0]/ZStack[0]/VStack[0]/Frame[0]/Frame[0]/HStack[0]/CheckBox[0]"
        )
        await cb.click()
        await update_stage_async()
        self.assertFalse(self.cb_wrapper.get_value())

        await cb.click()
        await update_stage_async()
        self.assertTrue(self.cb_wrapper.get_value())

        self.assertTrue(self.cb_callbacks == [False, True], "CheckBox callback function not working as expected")

        window.destroy()

    async def testColorPickerWrapper(self):
        # This test emulates clicking on the widget wrapper and choosing a new color through clicking and through API
        window_title = "UI_Widget_Wrapper_Test_Window_ColorPicker_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self._selected_colors = []

        def on_color_picked(color):
            # The chosen colors should be the same every time
            self._selected_colors.append(color)
            self.assertTrue(
                (color == [0.2545888423919678, 0.3593621551990509, 0.4641350507736206, 1.0] or color == [1, 1, 1, 1])
            )

        with window.frame:
            color_picker = ColorPicker("Color Picker", [0.5, 0.6, 0.7, 1], on_color_picked_fn=on_color_picked)
        await update_stage_async()

        picker = ui_test.find(f"{window_title}//Frame/Frame[0]/HStack[0]/ColorWidget[0]")
        await picker.click()

        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(400, 300))
        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(400, 600))
        await update_stage_async()

        color_picker.set_color(np.ones(4))
        self.assertTrue(np.all(color_picker.get_color() == np.ones(4)))
        self.assertTrue(len(self._selected_colors) == 2)

        window.destroy()

    async def testDropDownWrapperBasicFunctionality(self):
        # This test emulates selecting an item from a DropDown and setting one manually using the API
        window_title = "UI_Widget_Wrapper_Test_Window_DropDown_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.selections = ["A", "B", "C", "D"]
        self.sel_ind = 0

        def on_dropdown_selected(selection):
            self.assertTrue(selection == self.selections[min(self.sel_ind, len(self.selections) - 1)])
            self.sel_ind += 1

        def populate_fn():
            return self.selections

        with window.frame:
            with CollapsableFrame("", collapsed=False):
                dropdown = DropDown(
                    "DropDown", populate_fn=populate_fn, on_selection_fn=on_dropdown_selected, keep_old_selections=True
                )
        dropdown.repopulate()
        self.assertTrue(dropdown.get_selection() == "A")
        self.assertTrue(dropdown.get_selection_index() == 0)

        self.assertTrue(dropdown.get_items() == ["A", "B", "C", "D"])

        await update_stage_async()

        combobox = ui_test.find(
            f"{window_title}//Frame/CollapsableFrame[0]/Frame[0]/ZStack[0]/VStack[0]/Frame[0]/Frame[0]/HStack[0]/ComboBox[0]"
        )
        await combobox.click()
        await combobox.click()

        # Selects option B
        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(400, 165))

        await update_stage_async()
        self.assertTrue(dropdown.get_selection() == "B")
        self.assertTrue(dropdown.get_selection_index() == 1)

        dropdown.set_selection("C")
        self.assertTrue(dropdown.get_selection() == "C")

        dropdown.set_selection_by_index(3)
        self.assertTrue(dropdown.get_selection() == "D")

        window.destroy()

    async def testDropDownWrapperArticulationSelection(self):
        window_title = "UI_Widget_Wrapper_Test_Window_DropDown_Articulation_Selection_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self._no_path_ct = 0
        self._valid_path_ct = 0

        self._robot_path = "/ur10"

        create_new_stage()

        def on_articulation_selected(articulation_path):
            if articulation_path is None:
                self._no_path_ct += 1
            elif articulation_path == self._robot_path:
                self._valid_path_ct += 1
            else:
                self.assertTrue(False, "Invalid Articulation Selection")

        with window.frame:
            with CollapsableFrame("", collapsed=False):
                dropdown = DropDown("DropDown", on_selection_fn=on_articulation_selected, keep_old_selections=True)
                dropdown.set_populate_fn_to_find_all_usd_objects_of_type("articulation")

        dropdown.repopulate()
        dropdown.trigger_on_selection_fn_with_current_selection()
        self.assertTrue(self._no_path_ct == 1)

        self._timeline = omni.timeline.get_timeline_interface()

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/UR10/ur10.usd", self._robot_path)
        # Test that articulations are found both when the timeline is stopped and playing
        dropdown.repopulate()
        self.assertTrue(self._valid_path_ct == 1)

        self._timeline.play()
        dropdown.repopulate()
        self.assertTrue(self._valid_path_ct == 1)
        dropdown.trigger_on_selection_fn_with_current_selection()
        self.assertTrue(self._valid_path_ct == 2)

        delete_prim(self._robot_path)
        dropdown.repopulate()
        self.assertTrue(self._no_path_ct == 2)

    async def testFloatFieldWrapper(self):
        # This test emulates modifying a FloatField by dragging
        window_title = "UI_Widget_Wrapper_Test_Window_Float_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.last_value = 2.2
        self.max_value = 4.0
        self.min_value = -1.0
        self.step = 0.025

        self.value_changed_ct = 0

        def on_value_changed(value):
            self.value_changed_ct += 1
            value = np.round(value, decimals=2)
            self.assertTrue(value <= self.max_value)
            self.assertTrue(value >= self.min_value)

        with window.frame:
            with CollapsableFrame("", collapsed=False):
                float_field = FloatField(
                    "FloatField",
                    default_value=self.last_value,
                    step=self.step,
                    lower_limit=self.min_value,
                    upper_limit=self.max_value,
                    on_value_changed_fn=on_value_changed,
                )

        await ui_test.emulate_mouse_move(ui_test.Vec2(400, 130))
        await ui_test.emulate_mouse_drag_and_drop(ui_test.Vec2(400, 130), ui_test.Vec2(600, 130))
        await update_stage_async()

        self.assertTrue(
            float_field.get_value() == self.max_value, f"FloatField has an unexpected value: {float_field.get_value()}"
        )

        float_field.set_value(-3.0)
        self.assertTrue(float_field.get_value() == -1.0)

        self.max_value = 15.0
        self.min_value = 13.0

        float_field.set_upper_limit(15)
        float_field.set_lower_limit(13)
        self.assertTrue(float_field.get_value() == 13.0)

        self.assertTrue(float_field.get_lower_limit() == 13.0)
        self.assertTrue(float_field.get_upper_limit() == 15.0)

        self.assertTrue(self.value_changed_ct == 5)

        window.destroy()

    async def testIntFieldWrapper(self):
        # This test emulates modifying a IntField by dragging
        window_title = "UI_Widget_Wrapper_Test_Window_Int_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.last_value = 2
        self.max_value = 10
        self.min_value = -1

        self.value_changed_ct = 0

        def on_value_changed(value):
            self.value_changed_ct += 1
            self.assertTrue(value <= self.max_value)
            self.assertTrue(value >= self.min_value)

        with window.frame:
            with CollapsableFrame("", collapsed=False):
                int_field = IntField(
                    "IntField",
                    default_value=self.last_value,
                    lower_limit=self.min_value,
                    upper_limit=self.max_value,
                    on_value_changed_fn=on_value_changed,
                )

        await ui_test.emulate_mouse_move(ui_test.Vec2(400, 130))
        await ui_test.emulate_mouse_drag_and_drop(ui_test.Vec2(400, 130), ui_test.Vec2(600, 130))
        await update_stage_async()

        self.assertTrue(int_field.get_value() == self.max_value)

        int_field.set_value(-3.0)
        self.assertTrue(int_field.get_value() == -1)

        await update_stage_async()

        self.max_value = 15
        self.min_value = 13

        int_field.set_upper_limit(15)
        int_field.set_lower_limit(13)

        self.assertTrue(int_field.get_value() == 13)

        self.assertTrue(int_field.get_lower_limit() == 13)
        self.assertTrue(int_field.get_upper_limit() == 15)

        self.assertTrue(self.value_changed_ct == 6)

        window.destroy()

    async def testFrameWrapper(self):
        window_title = "UI_Widget_Wrapper_Test_Window_Frame_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.btn_fun_clicks = [0, 0]

        def build_fn():
            def on_click_fn():
                self.btn_fun_clicks[0] += 1

            btn = Button("Button", "BUTTON", on_click_fn=on_click_fn)

        with window.frame:
            frame = Frame(build_fn=build_fn)
        await update_stage_async()

        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(300, 120))
        await update_stage_async()

        def new_build_fn():
            def on_new_click_fn():
                self.btn_fun_clicks[1] += 1

            btn = Button("Button", "BUTTON", on_click_fn=on_new_click_fn)

        frame.set_build_fn(new_build_fn)
        await update_stage_async()
        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(300, 120))
        await update_stage_async()

        frame.enabled = False
        await update_stage_async()
        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(300, 120))
        await update_stage_async()

        self.assertTrue(np.all(self.btn_fun_clicks == [1, 1]))

        window.destroy()

    async def testScrollingFrameWrapper(self):
        window_title = "UI_Widget_Wrapper_Test_Window_ScrollingFrame_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.btn_fun_clicks = [0, 0]

        def build_fn():
            def on_click_fn():
                self.btn_fun_clicks[0] += 1

            btn = Button("Button", "BUTTON", on_click_fn=on_click_fn)

        with window.frame:
            frame = ScrollingFrame(build_fn=build_fn, num_lines=3)
        await update_stage_async()

        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(300, 120))
        await update_stage_async()

        frame.set_num_lines(4)

        def new_build_fn():
            def on_new_click_fn():
                self.btn_fun_clicks[1] += 1

            btn = Button("Button", "BUTTON", on_click_fn=on_new_click_fn)

        frame.set_build_fn(new_build_fn)
        await update_stage_async()
        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(300, 120))
        await update_stage_async()

        frame.enabled = False
        await update_stage_async()
        await ui_test.emulate_mouse_move_and_click(ui_test.Vec2(300, 120))
        await update_stage_async()

        self.assertTrue(np.all(self.btn_fun_clicks == [1, 1]))

        window.destroy()

    async def testStateButtonWrapper(self):
        # Width and height chosen such that Button is visible in UI Window
        window_title = "UI_Widget_Wrapper_Test_Window_StateButton_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.btn_clicks = [0, 0]

        def on_a_click_fn():
            self.btn_clicks[0] += 1

        def on_b_click_fn():
            self.btn_clicks[1] += 1

        with window.frame:
            state_button = StateButton(
                "StateButton", "A", "B", on_a_click_fn=on_a_click_fn, on_b_click_fn=on_b_click_fn
            )

        button = ui_test.find(f"{window_title}//Frame/Frame[0]/HStack[0]/Button[0]")
        await button.click()
        await update_stage_async()
        await button.click()
        await update_stage_async()

        self.assertTrue(self.btn_clicks == [1, 1])

        self.physics_step_count = 0

        def physics_step_count(step_size):
            self.physics_step_count += 1

        timeline = omni.timeline.get_timeline_interface()

        state_button.set_physics_callback_fn(physics_step_count)

        from isaacsim.core.api.world import World

        world = World()

        await world.initialize_simulation_context_async()

        timeline.play()
        await update_stage_async()

        await button.click()
        for i in range(6):
            await update_stage_async()

        self.assertTrue(self.physics_step_count == 9, "An unexpected number of physics step counts occured")
        self.assertTrue(self.btn_clicks == [2, 1])

        self.assertTrue(state_button.get_current_text() == "B")
        self.assertFalse(state_button.is_in_a_state())

        # Nothing should happen
        state_button.trigger_click_if_a_state()

        self.assertTrue(state_button.get_current_text() == "B")
        self.assertFalse(state_button.is_in_a_state())

        state_button.trigger_click_if_b_state()

        self.assertTrue(state_button.get_current_text() == "A")
        self.assertTrue(state_button.is_in_a_state())

        state_button.trigger_click_if_a_state()

        self.assertTrue(state_button.get_current_text() == "B")
        self.assertFalse(state_button.is_in_a_state())

        window.destroy()

    async def testStringFieldWrapper(self):
        # There is no easy way to rigorously test this one using ui_test
        # Significant functionality is missing from this test

        window_title = "UI_Widget_Wrapper_Test_Window_StringField_Test"
        width = 500
        height = 200
        window = await self._create_window(window_title, width, height)

        self.values = []

        def on_value_changed_fn(value: str):
            self.values.append(value)

        with window.frame:
            string_field = StringField("StringField", default_value="Text", on_value_changed_fn=on_value_changed_fn)

        string_field.set_value("New Text")

        self.assertTrue(self.values == ["New Text"])
        self.assertTrue(string_field.get_value() == "New Text")

        window.destroy()

    async def testTextBlockWrapper(self):
        # TextBlock does not allow user interaction, so this test is complete

        window_title = "UI_Widget_Wrapper_Test_Window_TextBlock_Test"
        width = 500
        height = 400
        window = await self._create_window(window_title, width, height)

        with window.frame:
            text_block = TextBlock("TextBlock", text="Text", num_lines=12, include_copy_button=True)

        text_block.set_text("New Text")
        self.assertTrue(text_block.get_text() == "New Text")

        window.destroy()

    async def testXYPlotWrapper(self):
        window_title = "UI_Widget_Wrapper_Test_Window_XYPlot_Test"
        width = 800
        height = 1000
        window = await self._create_window(window_title, width, height)

        with window.frame:
            with ui.VStack():
                # Test no errors with all default args
                plot1 = XYPlot("Plot 1")

                # Test no errors with all args given
                # Visually inspect that this plot is correct once per release
                plot2 = XYPlot(
                    "Plot 2",
                    x_data=[[1, 2, 3], [3.1, 3.7], np.arange(0, 4, 0.01)],
                    y_data=[[2, 3, 4], [1.1, -0.01], np.sin(np.arange(0, 4, 0.01))],
                    x_min=0.5,
                    y_min=0,
                    x_max=1.5,
                    y_max=2,
                    x_label="X Label",
                    y_label="Y Label",
                    plot_height=8,
                    show_legend=True,
                    legends=["L1", "L2"],
                    plot_colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                )

                # Test that auto-computed max and mins are correct
                plot3 = XYPlot(
                    "Plot 3",
                    x_data=[[1, 2, 3], [3.1, 3.7], np.arange(0, 4, 0.01)],
                    y_data=[[2, 3, 4], [1.1, -0.01], np.sin(np.arange(0, 4, 0.01))],
                    show_legend=False,
                    plot_height=8,
                )

                ui.Spacer()

        plot1.set_data([0, 4], [4, 0])

        self.assertTrue(plot1.get_x_min() == 0)
        self.assertTrue(plot1.get_y_min() == 0)
        self.assertTrue(plot1.get_x_max() == 4)
        self.assertTrue(plot1.get_y_max() == 4)

        self.assertTrue(plot2.get_legends() == ["L1", "L2", "F_2(x)"])
        self.assertTrue(plot2.get_plot_height() == 8)
        plot3.set_plot_height(4)
        self.assertTrue(plot3.get_plot_height() == 4)

        plot3.set_plot_visible_by_index(0, False)
        plot3.set_show_legend(True)

        # Use this line to conduct visual inspection of plots for correctness
        # await asyncio.sleep(30)

        window.destroy()
