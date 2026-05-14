import omni.ext
from omni.kit.window.property import get_window
from .my_property_delegate import MyFilteredSchemeDelegate
import asyncio

Reflectance_Binding_widget_NAME = "Reflectance_Binding"


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class RapidLibsoworldExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[rapid.Property] rapid Property startup")
        # 获取PropertyWindow实例
        window = get_window()
        if window:
            # 创建并注册自定义delegate
            my_delegate = MyFilteredSchemeDelegate()
            window.register_scheme_delegate(
                scheme="prim",  # 或其他的scheme名称
                name="my_filtered_delegate",
                delegate=my_delegate
            )

            # 设置delegate布局，确保自定义的delegate优先
            window.set_scheme_delegate_layout("prim", ["my_filtered_delegate"])

            # 将material_binding的widget标题改为reflectance_binding，next_update_async确保在下一帧执行
            self._task = omni.kit.app.get_app().next_update_async()
            asyncio.ensure_future(self._rename_material_binding_widget())

    def on_shutdown(self):
        print("[rapid.Property] rapid Property shutdown")

    async def _rename_material_binding_widget(self):
        """找到并重命名 material_binding widget"""
        await self._task  # 等待一帧

        # 1. 获取PropertyWindow单例
        window = get_window()
        if not window:
            # 如果第一次没获取到，可以稍作等待再试
            await asyncio.sleep(0.5)
            window = get_window()

        if not window:
            print("无法获取Property Window实例。")
            return

        # 2. 确定当前的scheme，通常对于Prim是"prim"
        target_scheme = "prim"

        # 3. 从TOP stack中查找名为"material_binding"的widget
        # 根据你之前打印的信息，它在TOP stack中
        top_widgets_dict = window._widgets_top.get(target_scheme)

        if not top_widgets_dict:
            print(f"在 scheme '{target_scheme}' 中未找到TOP stack的widget字典。")
            return

        target_widget = top_widgets_dict.get("material_binding")

        if not target_widget:
            print("未找到名为 'material_binding' 的widget。")
            # 可以打印出所有widget名称来调试
            print(f"可用的widget: {list(top_widgets_dict.keys())}")
            return

        # 4. 直接修改其 _title 属性
        # 这是最关键的一步，但依赖于该widget内部确实有这个属性
        if hasattr(target_widget, '_title'):
            old_title = target_widget._title
            target_widget._title = Reflectance_Binding_widget_NAME  # 修改为你想要的新标题
        else:
            # 如果 _title 属性不存在，尝试其他可能存储标题的属性
            print("该widget没有 '_title' 属性。尝试查找其他标题属性...")
            # 可以检查是否有 'title', 'name', 或 'label' 等属性
            for attr_name in ['title', 'name', 'label', '_label']:
                if hasattr(target_widget, attr_name):
                    setattr(target_widget, attr_name, Reflectance_Binding_widget_NAME)
                    break
            else:
                print(" 未找到可修改的标题属性。")
                # 作为最后的手段，可以尝试添加_title属性（不推荐，可能无效）
                target_widget._title = Reflectance_Binding_widget_NAME

        # 5. （可选）触发一次UI重建，使更改立即生效
        # 这在某些情况下是必要的，因为标题可能在构建时被缓存
        try:
            window.request_rebuild()
        except Exception as e:
            print(f"ℹ请求重建时出错（可能不影响效果）: {e}")