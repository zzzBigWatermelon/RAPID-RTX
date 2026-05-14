# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import carb

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
from isaacsim.gui.components.callbacks import (
    on_copy_to_clipboard,
    on_docs_link_clicked,
    on_open_folder_clicked,
    on_open_IDE_clicked,
)


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestUI(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        await omni.kit.app.get_app().next_update_async()
        pass

    # After running each test
    async def tearDown(self):
        await omni.kit.app.get_app().next_update_async()
        pass

    # Run for a single frame and exit
    async def test_ui(self):
        await omni.kit.app.get_app().next_update_async()
        pass

    # TODO: Disabling this test as it hangs on TC on shutdown
    # async def test_clipboard(self):
    #     import pyperclip

    #     on_copy_to_clipboard("test")
    #     try:
    #         self.assertEqual(pyperclip.paste(), "test")
    #     except pyperclip.PyperclipException:
    #         carb.log_warn(pyperclip.EXCEPT_MSG)
    #         return

    async def test_ide(self):
        import os

        on_open_IDE_clicked(os.path.dirname(__file__), __file__)

    # TODO: this test causes TC to hang on exit, disabling
    async def test_docs(self):
        # on_open_folder_clicked(os.path.dirname(__file__)) # TODO: this test fails on TC due to permissions
        on_docs_link_clicked("https://docs.omniverse.nvidia.com")
