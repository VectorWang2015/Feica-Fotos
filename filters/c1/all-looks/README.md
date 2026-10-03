# Feica Fotos · Leica Q Typ116 全套滤镜

这套配置包含21款滤镜，每款提供25、50、75、100四档强度。Steve McCurry和Greg Williams另外提供0档端点。六款单色滤镜各有红、橙、黄、绿、蓝五种滤色版本，同样提供四档强度。

共206份ICC。每个文件包含Q1相机基底与所选滤镜的颜色变换，使用时选择一份即可。

## 安装和使用

1. 解压到自己的滤镜备份目录。
2. 将需要使用的`.icm`文件复制到Capture One的相机ICC目录。Windows中常见的位置是安装目录下的`Color Profiles\Common`，例如`C:\Program Files\Capture One\Capture One\Color Profiles\Common`。保留该目录中的原有配置。
3. 重启Capture One，打开Q Typ116拍摄的RAW，为它创建一个克隆变体。
4. 在“基本特性 / Base Characteristics”的ICC列表选择名称以`Feica Fotos-Q1`开头的滤镜。
5. 先选定一个固定基础曲线，例如Film Standard，再调整曝光、白平衡和其他参数。原变体可以继续使用原来的Generic和Auto设置。

文件名中的`S025`、`S050`、`S075`、`S100`表示强度。带`FilterRed`等后缀的是单色滤色版本。需要恢复原相机颜色时，切回Leica Q Generic。

## 文件分类

- `colors`：13款彩色滤镜，共52份。
- `monochrome`：6款单色或调色单色滤镜，共24份。
- `artist`：Steve McCurry和Greg Williams，各5档，共10份。
- `attachments`：6款单色滤镜的五色附件，共120份。

Steve和Greg的0档各自带有低端风格。其他滤镜的25、50、75档按原图与完整效果混合，100档应用完整色表。单色100档保持色表给出的灰度或调色效果。

## 适用范围

配置以Capture One的Q Typ116 Generic为基底，面向Q1 RAW。颜色处理采用已恢复的滤镜表，并对超出工作色域的颜色作连续压缩。它们保留了原生33³网格、输入曲线和输出结构。

这些文件已完成本地数值、结构和LittleCMS检查。Capture One的基础曲线、白平衡和其他调整仍会影响最终画面；当前版本适合作为可调整的滤镜起点。

`MANIFEST.json`列出每个文件的滤镜、强度、附件和SHA256。`SHA256SUMS.txt`可用于核对文件完整性。厂商颜色数据和相机配置的相关权利归各自权利人所有。
