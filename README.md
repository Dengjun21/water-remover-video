# 🎬 视频去水印工具

一个功能强大的视频水印去除工具，支持 GUI 可视化操作和 CLI 命令行两种模式。提供多种水印去除算法，从简单的模糊遮盖到 AI 深度学习修复，满足不同场景需求。

## ✨ 功能特性

- **可视化框选**：在视频画面上直接拖拽标注水印区域，支持多区域
- **8 种去除算法**：从简单遮盖到 AI 修复，按需选择
- **动态水印跟踪**：帧差法自动检测移动水印位置，逐帧追踪
- **时间段控制**：为每个水印区域设置生效时间段
- **实时预览**：处理前可预览单帧效果，左右对比原图与修复结果
- **音频保留**：通过 ffmpeg 合并原始音频，输出完整视频
- **GPU 加速**：LaMa AI 修复支持 CUDA GPU 加速

## 🖥️ 截图

<!-- 可在此处添加截图 -->

## 📦 安装

### 环境要求

- Python 3.10+
- ffmpeg（系统安装或通过 `pip install imageio-ffmpeg`）
- NVIDIA GPU（可选，用于 LaMa AI 加速）

### 安装依赖

```bash
pip install opencv-python numpy pillow customtkinter imageio-ffmpeg
```

LaMa AI 修复（可选）：

```bash
pip install simple-lama-inpainting
```

CUDA GPU 加速（可选，推荐有 NVIDIA 显卡的用户）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 🚀 使用方法

### GUI 模式（推荐）

```bash
python watermark_remover_gui.py
```

**操作流程：**

1. 点击「📁 打开视频」加载视频文件
2. 在画面上拖拽框选水印区域（支持多个区域）
3. 在「水印区域」标签页调整区域参数（位置、大小、时间段、跟踪模式）
4. 在「去水印方法」标签页选择修复算法
5. 点击「👁 预览效果」查看当前帧处理结果（左原图 / 右处理后）
6. 满意后点击「🚀 开始处理」导出视频

### CLI 模式

```bash
python remove_watermark.py input.mp4 output.mp4 [选项]
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频路径 | 必填 |
| `output` | 输出视频路径 | 必填 |
| `--tl` | 左上角水印区域 `x,y,w,h` | `0,0,200,50` |
| `--br` | 右下角水印区域 `x,y,w,h` | `1030,665,250,55` |
| `--method` | 去水印方法 | `cover` |
| `--radius` | 修复半径 | `5` |

**示例：**

```bash
# 使用 Inpaint 修复
python remove_watermark.py input.mp4 output.mp4 --method inpaint --radius 8

# 指定水印区域 + LaMa AI 修复
python remove_watermark.py input.mp4 output.mp4 --tl 0,0,200,50 --br 1030,665,250,55 --method lama

# 裁剪模式
python remove_watermark.py input.mp4 output.mp4 --method crop
```

## 🔧 去水印算法对比

| 方法 | 速度 | 效果 | 说明 |
|------|------|------|------|
| **Inpaint (TELEA)** | ⚡⚡⚡ | ⭐⭐⭐ | 从水印边缘向内推算修复，速度快效果稳定（GUI 默认） |
| **NS (Navier-Stokes)** | ⚡⚡⚡ | ⭐⭐⭐ | 流体力学算法，适合平滑背景区域 |
| **贴片覆盖 (Cover)** | ⚡⚡⚡⚡ | ⭐⭐⭐ | 从外围取样填充 + 羽化混合，适合复杂纹理（CLI 默认） |
| **LaMa AI** | ⚡ (GPU) / 🐌 (CPU) | ⭐⭐⭐⭐⭐ | 深度学习修复，理解画面语义，效果最自然 |
| **高斯模糊** | ⚡⚡⚡⚡ | ⭐⭐ | 模糊遮盖，类似剪映的模糊效果 |
| **马赛克** | ⚡⚡⚡⚡ | ⭐ | 像素化遮盖 |
| **纯色遮盖** | ⚡⚡⚡⚡ | ⭐ | 黑色块覆盖 |
| **裁剪缩放** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 裁掉水印边缘后放大画面，最彻底但改变画面构图 |

### 与剪映模糊去水印的区别

剪映等工具只能在水印上叠加模糊或马赛克，水印痕迹依然可见，效果粗糙。本工具的 **Inpaint/NS** 算法从水印边缘向内推算修复，能还原被遮挡的背景；**LaMa AI** 则通过深度学习理解画面语义，生成最自然的修复结果。

## 📂 项目结构

```
social_media/
├── watermark_remover_gui.py   # GUI 主程序 (CustomTkinter)
├── remove_watermark.py        # CLI 命令行工具
├── WatermarkRemover.spec      # PyInstaller 打包配置
└── README.md
```

## 🛠️ 打包为 EXE

```bash
pip install pyinstaller
python -m PyInstaller WatermarkRemover.spec --noconfirm
```

生成的 EXE 文件位于 `dist/WatermarkRemover.exe`。

> **注意**：EXE 打包版本不包含 LaMa AI 修复（torch 体积过大）。如需使用 LaMa，请直接运行 Python 源码。

## 📝 技术细节

- **GUI 框架**：[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - 现代 dark 主题 UI
- **视频处理**：OpenCV + ffmpeg
- **AI 修复**：[LaMa](https://github.com/advimman/lama) via [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- **动态跟踪**：帧差法 + 轮廓检测，逐帧定位移动水印
- **LaMa 优化**：半分辨率 patch 处理，GPU 模式下 241 帧 720p 视频约 46 秒

## 📄 License

MIT
