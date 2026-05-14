# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [1.0.0] - 2026-4-30
### 新增

- [rapid.ExampleScene]新增简单森林场景和简单动态风力场.
- [rapid.LiDAR]新增星载大光斑激光雷达模拟功能，输出能量分布txt，png图片和用于模拟波形的点云。
- [rapid.Tool]新增多光谱hdr图像和las点云数据的查看器.

### 变更

- isaac-sim5.1版本将Kit SDK 中之前名为 omni.usd.schema.forcefield 的扩展已被名为 PhysxForceFieldAPI 的新 API 模式取代.
- [rapid.Simulation]改为float16的辐亮度输出，取消了曝光调整。
- [rapid.LiDAR]取消原版RTXLiDAR的世界坐标输出（点云大范围漂移），改用输出LiDAR的局部点云数据+世界转换矩阵后拼接输出世界坐标点云。
- [rapid.Utility]统一使用Utility中的combo_box的复选框UI的逻辑代码。

### 修复

- 视口闪烁问题修复，VULKAN图形API导致老显卡出现渲染时序问题。已在apps\isaacsim.exp.full.kit中设置vulkan = false，即启动D3D12图形API。
- 点开菜单栏控制的窗口出现KeyError: 'Window_Browsers'的窗口回收问题，[rapid.Layout]的extension.py中添加target._refresh_menu_item = safe_refresh替换系统的safe_refresh，去掉报错提示。
- [rapid.Layout]修改了菜单栏file下的功能显示顺序。

### 安全

- None

### 已知问题

- [rapid.LiDAR]星载大光斑激光雷达模拟没有添加噪声算法，是平滑的能量曲线。
- [rapid.LiDAR]地基LiDAR的模拟总是会缺少最后的两帧数据，异步代码的问题
- [rapid.SceneConstruction]SceneConstruction窗口下，光照中心的输入没有作用，已经采用默认计算世界bbox做为光照中心。没有光源的强度参数控制，只能通过光源的property控制强度。
- [rapid.ExampleScenes]wytham场景的叶片透射率是红光波段的，场景从下向上看是红色叶片。
- [rapid.ReflectanceDarabase]ReflectanceDarabase窗口下的plot按钮未绑定实际功能，选中的反射率表格数据的被画成光谱折线。
