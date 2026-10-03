# Feica Fotos

Feica Fotos 是一个离线照片滤镜 App，支持 Ubuntu 和 Windows。打开照片、选择滤镜、调整强度，再导出 PNG 或 JPEG。

App 提供 21 款滤镜和 5 种单色滤色附件，强度可按 0.01 调整。DNG 输入使用文件中的内嵌 JPEG 预览，导出保留预览的完整尺寸。

## 构建 App

使用 Python 3.12，在仓库根目录执行以下命令。首次安装依赖需要访问 Python 包源。

### Windows

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-app.txt -r requirements-build.txt
.venv\Scripts\python.exe scripts\build_feica_fotos.py --with-local-resources filters\looks
```

构建完成后运行 `dist\Feica Fotos\Feica Fotos.exe`。用户已成功编译Windows 0.3版，本次0.3.1沿用同一构建流程。

### Ubuntu

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-app.txt -r requirements-build.txt
.venv/bin/python scripts/build_feica_fotos.py --with-local-resources filters/looks
```

构建完成后运行 `"./dist/Feica Fotos/Feica Fotos"`。

Windows 和 Linux 分别在对应系统上打包。分发时保留整个 `dist/Feica Fotos` 文件夹，里面包含运行库和滤镜资源。重新构建前，将已有输出移到备份目录。

开发时可以直接运行源码：

```bat
:: Windows
launch-feica-fotos.cmd --resource-dir filters\looks
```

```bash
# Ubuntu
./launch-feica-fotos.sh --resource-dir filters/looks
```

## 在 App 中使用滤镜

1. 打开 JPG、DNG 或 RGB PNG。
2. 在颜色、单色或 Artist 分组中选择滤镜，拖动滑块或直接输入强度。
3. 按 `B` 对照原图，按 `Ctrl+I` 查看滤镜介绍。
4. 按 `Ctrl+Shift+S` 导出副本，选择一个新文件名。

单色滤镜的“滤色”菜单提供红、橙、黄、绿、蓝选项。Steve McCurry 的零档对应低强度风格端点；需要完全未处理的画面时选择“原图”。

介绍中的 preview 标记表示当前参数规则仍在完善。源照片保持只读，导出文件使用 sRGB。

## 在 Capture One 中使用滤镜

### Q1 RAW：全套相机ICC

[filters/c1/all-looks](filters/c1/all-looks) 提供21款滤镜的25、50、75、100四档强度，Steve McCurry和Greg Williams另有0档端点。六款单色滤镜各有五种滤色附件和四档强度，共206份ICC。

- `colors`：13款彩色滤镜，共52份。
- `monochrome`：6款单色或调色单色滤镜，共24份。
- `artist`：Steve McCurry和Greg Williams，共10份。
- `attachments`：单色滤镜的红、橙、黄、绿、蓝版本，共120份。

使用步骤：

1. 为Q1 RAW创建一个克隆变体，保留原来的Generic配置。
2. 将需要的`.icm`文件加入Capture One的自定义ICC配置。Windows安装中常见的位置是安装目录下的`Color Profiles\Common`；例如`C:\Program Files\Capture One\Capture One\Color Profiles\Common`。复制时保留已有文件。
3. 重启Capture One，在“基本特性 / Base Characteristics”的ICC列表选择`Feica Fotos-Q1`开头的滤镜。
4. 选择固定的基础曲线，例如Film Standard，再调整曝光和白平衡。原来的Auto工作流可以继续保留在原变体中。

文件名中的`S025`等数字表示强度，`FilterRed`等后缀表示滤色附件。每次选择一份ICC即可；需要恢复原相机颜色时切回Generic。

这套配置保留Q Typ116相机基底，使用完整滤镜颜色变换，并对超出工作色域的颜色作连续压缩。单色100档使用色表给出的灰度或调色效果。当前版本已完成本地数值和LittleCMS检查，C1的曲线和其他调整仍会影响最终画面。

每份文件的参数和SHA256保存在集合内的`MANIFEST.json`中。

### 已显影的 sRGB TIFF：RGB 滤镜组

[filters/rendered-srgb](filters/rendered-srgb) 保存了 18 组滤镜，每组有 25%、50%、75%、100% 四档，共 72 个 ICC 和 72 个配套样式。

1. 将 `ICC` 中的文件加入 Capture One 的 ICC 配置目录。
2. 在“样式和预设 → 导入样式”中导入 `Styles` 里的 `.costyle` 文件。
3. 打开已显影的 sRGB TIFF，在用户样式中选择滤镜和强度。

样式负责选择对应 ICC，曝光、曲线和白平衡沿用当前设置。名称带 `MonoBase` 的四组适用于已有的单色底图。导出时使用正常的输出色彩配置，例如 sRGB。

同目录的 `CUBE` 可供支持三维 LUT 的调色软件使用。App 构建使用的色表位于 [filters/looks](filters/looks)。

[filters/diagnostics](filters/diagnostics) 中的 Bypass 用于比较相机基底，OrderProbe 用于检查颜色变换顺序，供开发时做对照。
