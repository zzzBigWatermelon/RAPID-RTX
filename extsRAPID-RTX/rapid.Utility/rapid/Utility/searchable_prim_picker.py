import omni.ui as ui
import omni.usd
from pxr import Usd, UsdGeom, UsdLux
import asyncio


class SearchablePrimPicker(ui.Window):
    def __init__(self, title="Select Prim", type_filter=UsdGeom.Camera, on_select_fn=None):
        """
        Args:
            title: 窗口标题
            type_filter: 过滤类型。支持：
                         1. 单个类型: UsdGeom.Camera
                         2. 类型列表: [UsdGeom.Camera, UsdGeom.Mesh, UsdLux.Light]
                         3. None: 不过滤，显示所有
            on_select_fn: 选中回调
        """
        super().__init__(title, width=400, height=500, visible=True)
        self._type_filter = type_filter
        self._on_select_fn = on_select_fn
        self._all_prims = []

        # 定义需要过滤掉的默认路径
        self._excluded_paths = [
            '/Environment',
            "/OmniverseKit_Persp",
            "/OmniverseKit_Front",
            "/OmniverseKit_Top",
            "/OmniverseKit_Right",
            "/Omniversekit_Persp"
        ]

        self._build_ui()
        self._scan_stage()
        self._refresh_list()

    def _build_ui(self):
        with self.frame:
            with ui.VStack(spacing=10, padding=15):
                with ui.HStack(height=0):
                    ui.Label("Search:", width=50)
                    self._search_field = ui.StringField()
                    self._search_field.model.add_value_changed_fn(lambda m: self._refresh_list())

                ui.Line(style={"color": 0x33FFFFFF}, height=2)

                with ui.ScrollingFrame(
                    height=ui.Fraction(1),
                    style={"background_color": 0xFF1A1A1A, "border_radius": 5}
                ):
                    self._list_stack = ui.VStack(spacing=2, padding=5)

                with ui.HStack(height=0):
                    ui.Spacer(width=5)
                    ui.Button("Close", clicked_fn=self._close_window, width=ui.Fraction(1))
                    ui.Spacer(width=5)

    def _scan_stage(self):
        """扫描舞台并根据单类型或多类型进行过滤"""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        self._all_prims = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())

            # 1. 首先检查是否在排除路径列表中（如果是排除路径的子节点也过滤掉）
            if any(path.startswith(ex) for ex in self._excluded_paths):
                continue

            # 2. 判断类型逻辑
            is_match = False

            if self._type_filter is None:
                # 情况 A: 不限制类型
                is_match = True
            elif isinstance(self._type_filter, (list, tuple)):
                # 情况 B: 传入的是列表或元组，只要符合其中一个类型即可
                # 使用 any() 配合 prim.IsA()
                is_match = any(prim.IsA(t) for t in self._type_filter)
            else:
                # 情况 C: 传入的是单个类型
                is_match = prim.IsA(self._type_filter)

            # 3. 如果匹配，加入列表
            if is_match:
                self._all_prims.append(path)

    def _refresh_list(self):
        """更新 UI 列表"""
        search_text = self._search_field.model.as_string.strip().lower()
        self._list_stack.clear()

        found_count = 0 

        with self._list_stack:
            for path in self._all_prims:
                if not search_text or search_text in path.lower():
                    found_count += 1
                    ui.Button(
                        path, 
                        height=25, 
                        clicked_fn=lambda p=path: self._on_item_selected(p),
                        style={
                            "text_align": ui.Alignment.LEFT,
                            "background_color": 0x00000000,
                            "margin": 2
                        }
                    )

            if found_count == 0:
                with ui.VStack(height=40):
                    ui.Spacer(height=10)
                    ui.Label(
                        "No matching prims found", 
                        style={"color": 0xFF666666, "font_style": "italic"},
                        alignment=ui.Alignment.CENTER
                    )

    def _on_item_selected(self, path):
        if self._on_select_fn:
            self._on_select_fn(path)
        self._close_window()

    def _close_window(self, *args):
        self.visible = False
        async def delayed_destroy():
            await asyncio.sleep(0.1)
            self.destroy()
        asyncio.ensure_future(delayed_destroy())