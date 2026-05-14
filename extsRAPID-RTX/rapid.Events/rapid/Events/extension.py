import omni.ext

# 定义事件名称
REQUEST_SIMULATION_START = "rapid.simulation:REQUEST_START"  # 模拟开始事件发出
SIMULATION_PARAMS_READY = "rapid.simulation:PARAMS_READY"  # 窗口参数收集事件
CONFIG_REFRESHED = "rapid.events.CONFIG_REFRESHED"  # 窗口模拟参数更新事件


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class EventsExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[rapid.Events] rapid Events startup")

    def on_shutdown(self):
        print("[rapid.Events] rapid Events shutdown")
