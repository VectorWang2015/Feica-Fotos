# Feica Fotos

Local Looks 是一个离线照片滤镜 App，支持 Ubuntu 和 Windows。打开照片、选择滤镜、调整强度，再导出 PNG 或 JPEG。

App 提供 21 款滤镜和 5 种单色滤色附件，强度可按 0.01 调整。DNG 输入使用文件中的内嵌 JPEG 预览，导出保留预览的完整尺寸。

## 构建 App

使用 Python 3.12，在仓库根目录执行以下命令。首次安装依赖需要访问 Python 包源。

### Windows

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-app.txt -r requirements-build.txt
.venv\Scripts\python.exe scripts\build_local_looks.py --with-local-resources filters\looks
```

构建完成后运行 `dist\LocalLooks\LocalLooks.exe`。

### Ubuntu

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-app.txt -r requirements-build.txt
.venv/bin/python scripts/build_local_looks.py --with-local-resources filters/looks
```

构建完成后运行 `./dist/LocalLooks/LocalLooks`。

Windows 和 Linux 分别在对应系统上打包。分发时保留整个 `dist/LocalLooks` 文件夹，里面包含运行库和滤镜资源。重新构建前，将已有输出移到备份目录。

开发时可以直接运行源码：

```bat
:: Windows
launch-local-looks.cmd --resource-dir filters\looks
```

```bash
# Ubuntu
./launch-local-looks.sh --resource-dir filters/looks
```

## 在 App 中使用滤镜

1. 打开 JPG、DNG 或 RGB PNG。
2. 在颜色、单色或 Artist 分组中选择滤镜，拖动滑块或直接输入强度。
3. 按 `B` 对照原图，按 `Ctrl+I` 查看滤镜介绍。
4. 按 `Ctrl+Shift+S` 导出副本，选择一个新文件名。

单色滤镜的“滤色”菜单提供红、橙、黄、绿、蓝选项。Steve McCurry 的零档对应低强度风格端点；需要完全未处理的画面时选择“原图”。

介绍中的 preview 标记表示当前参数规则仍在完善。源照片保持只读，导出文件使用 sRGB。

## 在 Capture One 中使用滤镜

### Q1 RAW：Vivid Preview

使用 [filters/c1](filters/c1) 中的 `LeicaQTyp116-LocalLooks-VividPreview-Native33-v1.icm`。这个配置在 Q Typ116 的原生相机配置上叠加了较温和的 Vivid 效果，目前处于实验阶段。

1. 为 Q1 RAW 创建一个克隆变体，保留原来的 Generic 配置。
2. 将 `.icm` 文件加入 Capture One 的自定义 ICC 配置。Windows 安装中常见的位置是安装目录下的 `Color Profiles\Common`；例如 `C:\Program Files\Capture One\Capture One\Color Profiles\Common`。复制时保留已有文件。
3. 重启 Capture One，在“基本特性 / Base Characteristics”的 ICC 列表选择 `LocalLooks-Q1-VividPreview-Native33-v1`。
4. 选择固定的基础曲线，例如 Film Standard，再调整曝光和白平衡。原来的 Auto 工作流可以继续保留在原变体中。

这一款只需选择一个 ICC。切回 Generic 即可恢复原相机配置。

### 已显影的 sRGB TIFF：RGB 滤镜组

[filters/rendered-srgb](filters/rendered-srgb) 保存了 18 组滤镜，每组有 25%、50%、75%、100% 四档，共 72 个 ICC 和 72 个配套样式。

1. 将 `ICC` 中的文件加入 Capture One 的 ICC 配置目录。
2. 在“样式和预设 → 导入样式”中导入 `Styles` 里的 `.costyle` 文件。
3. 打开已显影的 sRGB TIFF，在用户样式中选择滤镜和强度。

样式负责选择对应 ICC，曝光、曲线和白平衡沿用当前设置。名称带 `MonoBase` 的四组适用于已有的单色底图。导出时使用正常的输出色彩配置，例如 sRGB。

同目录的 `CUBE` 可供支持三维 LUT 的调色软件使用。App 构建使用的色表位于 [filters/looks](filters/looks)。

[filters/diagnostics](filters/diagnostics) 中的 Bypass 用于比较相机基底，OrderProbe 用于检查颜色变换顺序，供开发时做对照。
