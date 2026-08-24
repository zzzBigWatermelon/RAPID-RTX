import omni.ext

class RapidUnitreeLidarL2Extension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[rapid.unitree_lidarl2] rapid unitree_lidarl2 startup")

    def on_shutdown(self):
        print("[rapid.unitree_lidarl2] rapid unitree_lidarl2 shutdown")
