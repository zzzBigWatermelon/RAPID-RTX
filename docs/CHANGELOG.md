# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-8-26

### 新增

- [rapid.chat_rapid]新增 RAPID AI 系统,该系统基于 LangChain 框架，通过自然语言交互即可快速构建复杂的森林场景。其核心由场景构建 Agent 驱动，能够自动编排工具链、组合多种 Tools，并支持自定义参数，实现高效、智能的 3D 场景生成。

### 变更

- [rapid.SceneConstruction]重构直射光的光源模型,将太阳直射光模型从圆盘光更新为平行光，并设定太阳角为 0.53°，以提升场景光照的真实感。
- [rapid.SceneConstruction]光源系统升级，标定了光源强度与地面辐照度的映射曲线，新增散射光能量占比的 UI 调节入口，实现光照能量控制。

### 修复

- [rapid.ReflectanceDatabase(spectral_preview_window.py)]修复 CSV 文件创建路径与源文件冲突导致反射率数据无法导入的问题
- [rapid.Observation]优化了正射相机配置逻辑：用户仅需输入宽度范围，系统将根据图像宽高比自动计算对应的范围。
- [rapid.ExampleScenes]移除 ExampleScenes 扩展以减小存储占用，所有示例场景已迁移至GitHub的USD-Scenes仓库中。
- [rapid.ReflectanceDatabase] 优化表格编辑交互：将双击进入编辑状态改为单击触发，提升操作灵敏度与响应速度。 
- [rapid.ReflectanceDatabase] 限制 Name 列为只读：该列值与 CSV 文件名强绑定，禁止修改以防止数据不一致。
- [rapid.Observation]修复观测目标位置缺失的问题（之前固定为原点，现改为实际目标位置）。修复正射相机（Orthographic）的观测范围、航线计算与定标距离计算逻辑（此前错误地复用了透视相机的计算方式，现已独立适配）。

### 已知问题

- [rapid.ReflectanceDatabase]传感器设定触发波段长度不匹配警告，新建工程后，若在未执行反射率刷新（refresh）操作的情况下直接进行传感器设定，系统会弹出黄色警告提示：

  ```text
  Band length mismatch detected. Auto padded to 5.
  ```

  该警告源于波段一致性检查机制。经 Simulation 代码诊断确认，当前场景中检测到 5 个波段，但反射率数据仅有 4 组，二者数量不匹配，触发自动补齐至 5 个波段并产生警告。

- [rapid.LiDAR]星载大光斑激光雷达模拟没有添加噪声算法，是平滑的能量曲线。
- [rapid.LiDAR]地基LiDAR的模拟总是会缺少最后的两帧数据，异步代码的问题
- [rapid.ExampleScenes]wytham场景的叶片透射率是红光波段的，场景从下向上看是红色叶片。
- [rapid.ReflectanceDarabase]ReflectanceDarabase窗口下的plot按钮未绑定实际功能，选中的反射率表格数据的被画成光谱折线。


## [0.1.0] - 2026-4-30
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
