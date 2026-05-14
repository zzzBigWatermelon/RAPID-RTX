# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


from collections import namedtuple
from typing import Callable, List, Optional

import carb
import omni
import omni.ui as ui
from isaacsim.gui.components.ui_utils import BUTTON_WIDTH, LABEL_WIDTH, get_style
from omni.kit.property.usd.relationship import RelationshipTargetPicker
from omni.kit.window.popup_dialog.dialog import get_field_value


class DynamicComboBoxItem(ui.AbstractItem):
    def __init__(self, text):
        super().__init__()
        self.model = ui.SimpleStringModel(text)


class DynamicComboBoxModel(ui.AbstractItemModel):
    def __init__(self, args):
        super().__init__()

        self._current_index = ui.SimpleIntModel()
        self._current_index.add_value_changed_fn(lambda a: self._item_changed(None))
        self._items = []
        for i in range(len(args)):
            self._items.append(DynamicComboBoxItem(args[i]))

    def get_item_children(self, item):
        return self._items

    def get_item_value_model(self, item: ui.AbstractItem = None, column_id: int = 0):
        if item is None:
            return self._current_index
        return item.model

    def set_item_value_model(self, item: ui.AbstractItem = None, column_id: int = 0):
        self._current_index = item
        self._item_changed(None)
        self._current_index.add_value_changed_fn(lambda a: self._item_changed(None))


class SelectPrimWidget:
    """
    modeled after FormWidget (from omni.kit.window.popup_dialog.form_dialog) to add a widget that opens relationship selector
    """

    def __init__(self, label: str = None, default: str = None, tooltip: str = ""):
        self._label = label
        self._default_path = default
        self._tooltip = tooltip

        self._build_ui()

    def _build_ui(self):
        with ui.HStack(height=0):
            ui.Label(
                self._label,
                width=ui.Percent(29),
                style_type_name_override="Field.Label",
                word_wrap=True,
                name="prefix",
                tooltip=self._tooltip,
            )
            ui.Spacer(width=ui.Percent(1))
            path_model = ui.SimpleStringModel()
            path_model.set_value(self._default_path)
            self._prim = ui.StringField(path_model, width=ui.Percent(50))
            ui.Spacer(width=ui.Percent(1))
            ui.Button("Add", width=ui.Percent(19), clicked_fn=self._on_select_prim)

    def _on_select_prim(self):
        stage = omni.usd.get_context().get_stage()
        additional_widget_kwargs = {"target_name": "Prim"}
        self.stage_picker = RelationshipTargetPicker(
            stage,
            [],
            None,
            additional_widget_kwargs,
        )
        self.stage_picker.show(1, on_targets_selected=self._on_target_selected)

    def _on_target_selected(self, paths):
        self._prim.model.set_value(paths[0])

    def get_value(self):
        return self._prim.model.get_value_as_string()

    def destroy(self):
        self._prim = None


class ParamWidget:
    """
        modified FormWidget (from omni.kit.window.popup_dialog.form_dialog) to better format for parameter collection use

    Note:
        ParamWidget.FieldDef:
            A namedtuple of (name, label, type, default value) for describing the input field,
            e.g. FormDialog.FieldDef("name", "Name:  ", omni.ui.StringField, "Bob").

    """

    FieldDef = namedtuple("FormDialogFieldDef", "name label type default tooltip focused", defaults=["", False])

    def __init__(self, field_def: FieldDef):
        self._field = None
        self._build_ui(field_def)

    def _build_ui(self, field_def):
        with ui.HStack(height=0):
            ui.Label(
                field_def.label,
                width=ui.Percent(29),
                style_type_name_override="Field.Label",
                word_wrap=True,
                name="prefix",
                tooltip=field_def.tooltip,
            )
            ui.Spacer(width=ui.Percent(1))
            self._field = field_def.type(width=ui.Percent(70))
            if "set_value" in dir(self._field.model):
                self._field.model.set_value(field_def.default)

    def get_value(self):
        return get_field_value(self._field)

    def destroy(self):
        self._field = None
