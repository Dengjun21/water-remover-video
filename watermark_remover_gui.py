"""
豆包AI视频去水印工具 - GUI 版本 (CustomTkinter)
支持在视频画面上手动标注固定/动态水印区域，可视化操作。

用法: python watermark_remover_gui_v2.py
"""

import os
import sys
import shutil
import subprocess
import threading
import json

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

_lama_instance = None


def get_lama():
    """懒加载 LaMa 模型，首次调用时初始化"""
    global _lama_instance
    if _lama_instance is None:
        try:
            from simple_lama_inpainting import SimpleLama
            _lama_instance = SimpleLama()
        except ImportError:
            raise RuntimeError(
                "LaMa 模型未安装，请运行: pip install simple-lama-inpainting"
            )
    return _lama_instance


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def get_video_info(ffmpeg, video_path):
    cmd = [ffmpeg, "-i", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    info = {"width": 0, "height": 0, "fps": 24, "duration": 0}
    for line in (result.stderr or "").split("\n"):
        line = line.strip()
        if "Stream" in line and "Video" in line:
            for token in line.split(","):
                token = token.strip()
                if "x" in token:
                    parts = token.split("x")
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        info["width"] = int(parts[0].strip())
                        h_str = parts[1].strip().split()[0]
                        if h_str.isdigit():
                            info["height"] = int(h_str)
                if "fps" in token and token.split()[0].replace(".", "").isdigit():
                    info["fps"] = float(token.split()[0])
        if "Duration" in line:
            dur = line.split("Duration:")[1].split(",")[0].strip()
            parts = dur.split(":")
            if len(parts) == 3:
                info["duration"] = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return info


class VideoCanvas(tk.Canvas):
    """可框选区域的视频画布"""

    def __init__(self, parent, on_region_selected=None, **kwargs):
        super().__init__(parent, bg="#0a0a0f", highlightthickness=0, **kwargs)
        self.on_region_selected = on_region_selected
        self.photo = None
        self.image_id = None
        self.scale = 1.0
        self.display_w = 1
        self.display_h = 1
        self.video_w = 1
        self.video_h = 1
        self.offset_x = 0
        self.offset_y = 0

        self._start_x = 0
        self._start_y = 0
        self._rect_id = None
        self._drawing = False

        self.regions = []
        self._overlay_ids = []

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set_frame(self, cv_frame):
        """显示一帧画面 (BGR numpy array)"""
        rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.video_w = w
        self.video_h = h

        canvas_w = max(self.winfo_width(), 1)
        canvas_h = max(self.winfo_height(), 1)
        self.scale = min(canvas_w / w, canvas_h / h)
        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)
        self.offset_x = (canvas_w - self.display_w) // 2
        self.offset_y = (canvas_h - self.display_h) // 2

        resized = cv2.resize(rgb, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil_img)

        if self.image_id is None:
            self.image_id = self.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)
        else:
            self.coords(self.image_id, self.offset_x, self.offset_y)
            self.itemconfig(self.image_id, image=self.photo)

        self._redraw_overlays()

    def to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转为原图坐标"""
        img_x = int((canvas_x - self.offset_x) / self.scale)
        img_y = int((canvas_y - self.offset_y) / self.scale)
        img_x = max(0, min(img_x, self.video_w))
        img_y = max(0, min(img_y, self.video_h))
        return img_x, img_y

    def to_canvas_coords(self, img_x, img_y):
        """将原图坐标转为画布坐标"""
        return int(self.offset_x + img_x * self.scale), int(self.offset_y + img_y * self.scale)

    def _on_press(self, event):
        self._start_x = event.x
        self._start_y = event.y
        self._drawing = True
        if self._rect_id:
            self.delete(self._rect_id)
        self._rect_id = self.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#e94560", width=2, dash=(4, 2)
        )

    def _on_drag(self, event):
        if self._drawing and self._rect_id:
            self.coords(self._rect_id, self._start_x, self._start_y, event.x, event.y)

    def _on_release(self, event):
        if not self._drawing:
            return
        self._drawing = False
        if self._rect_id:
            self.delete(self._rect_id)
            self._rect_id = None

        x1 = min(self._start_x, event.x)
        y1 = min(self._start_y, event.y)
        x2 = max(self._start_x, event.x)
        y2 = max(self._start_y, event.y)

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        img_x1, img_y1 = self.to_image_coords(x1, y1)
        img_x2, img_y2 = self.to_image_coords(x2, y2)

        if self.on_region_selected:
            self.on_region_selected(img_x1, img_y1, img_x2 - img_x1, img_y2 - img_y1)

    def set_regions(self, regions):
        """设置要显示的水印区域列表"""
        self.regions = regions
        self._redraw_overlays()

    def _redraw_overlays(self):
        for oid in self._overlay_ids:
            self.delete(oid)
        self._overlay_ids.clear()

        for i, region in enumerate(self.regions):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cx1, cy1 = self.to_canvas_coords(x, y)
            cx2, cy2 = self.to_canvas_coords(x + w, y + h)

            color = "#e94560" if region["type"] == "fixed" else "#3a86ff"
            rid = self.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2)
            self._overlay_ids.append(rid)

            label = f'#{i+1} {region["type"]}'
            tid = self.create_text(cx1 + 4, cy1 + 4, anchor=tk.NW, text=label,
                                   fill=color, font=("Arial", 9, "bold"))
            self._overlay_ids.append(tid)


class WatermarkRemoverApp(ctk.CTk):
    _METHODS = [
        ("inpaint", "Inpaint 修复", "TELEA算法，从边缘向内修复，速度快效果稳定"),
        ("ns", "NS 修复", "Navier-Stokes流体算法，适合平滑背景区域"),
        ("cover", "贴片覆盖", "从水印外围取像素填充+羽化，适合复杂纹理"),
        ("lama", "LaMa AI 修复", "深度学习修复，效果最佳，需GPU加速"),
        ("blur", "高斯模糊", "模糊遮盖，类似剪映的模糊效果"),
        ("mosaic", "马赛克", "像素化遮盖"),
        ("block", "纯色遮盖", "黑色块覆盖"),
        ("crop", "裁剪缩放", "裁掉水印边缘后放大画面"),
    ]

    def __init__(self):
        super().__init__()
        self.title("豆包AI视频去水印工具")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.ffmpeg = find_ffmpeg()
        self.video_path = None
        self.cap = None
        self.video_info = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.current_frame = None

        self.regions = []
        self.selected_region_idx = None

        self.processing = False
        self._seeking = False
        self._updating_sliders = False

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- 顶部栏 ----
        top = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color=("gray90", "gray10"))
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)

        title_label = ctk.CTkLabel(top, text="🎬 豆包AI视频去水印工具",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(side=tk.LEFT, padx=20, pady=10)

        top_right = ctk.CTkFrame(top, fg_color="transparent")
        top_right.pack(side=tk.RIGHT, padx=20)

        self.btn_open = ctk.CTkButton(top_right, text="📁 打开视频", width=120,
                                       command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_video = ctk.CTkLabel(top_right, text="未选择视频文件",
                                       text_color="gray50", font=ctk.CTkFont(size=12))
        self.lbl_video.pack(side=tk.LEFT)

        # ---- 主区域 ----
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # 左侧：视频画布
        left = ctk.CTkFrame(main, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        canvas_frame = tk.Frame(left, bg="#0a0a0f")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.canvas = VideoCanvas(canvas_frame, on_region_selected=self._on_region_selected,
                                  width=800, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 时间线
        timeline = ctk.CTkFrame(left, fg_color="transparent")
        timeline.grid(row=1, column=0, sticky="ew", padx=8, pady=(8, 8))

        self.btn_prev = ctk.CTkButton(timeline, text="◀", width=36, command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_next = ctk.CTkButton(timeline, text="▶", width=36, command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=(0, 12))

        self.scale_timeline = ctk.CTkSlider(timeline, from_=0, to=100, command=self.on_timeline_seek)
        self.scale_timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_frame_info = ctk.CTkLabel(timeline, text="0 / 0", width=120,
                                            font=ctk.CTkFont(size=11))
        self.lbl_frame_info.pack(side=tk.LEFT, padx=(12, 0))

        # 右侧面板
        right = ctk.CTkFrame(main, width=380, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(right, width=380)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.tabview.add("水印区域")
        self.tabview.add("去水印方法")
        self.tabview.add("使用说明")

        self._build_regions_tab(self.tabview.tab("水印区域"))
        self._build_method_tab(self.tabview.tab("去水印方法"))
        self._build_help_tab(self.tabview.tab("使用说明"))

        # ---- 底部操作栏 ----
        bottom = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=("gray90", "gray10"))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_propagate(False)

        self.btn_preview = ctk.CTkButton(bottom, text="👁 预览效果", width=120,
                                          command=self.preview_result)
        self.btn_preview.pack(side=tk.LEFT, padx=(20, 8), pady=12)

        self.btn_process = ctk.CTkButton(bottom, text="🚀 开始处理", width=140,
                                          font=ctk.CTkFont(size=15, weight="bold"),
                                          command=self.start_process)
        self.btn_process.pack(side=tk.LEFT, padx=(0, 12))

        self.progress = ctk.CTkProgressBar(bottom, width=300)
        self.progress.pack(side=tk.LEFT, padx=(0, 12))
        self.progress.set(0)

        self.lbl_progress = ctk.CTkLabel(bottom, text="就绪", font=ctk.CTkFont(size=12))
        self.lbl_progress.pack(side=tk.LEFT)

        self.lbl_status = ctk.CTkLabel(bottom, text="", text_color="gray50",
                                        font=ctk.CTkFont(size=11))
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

    def _build_regions_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # 区域列表
        list_frame = ctk.CTkFrame(tab, fg_color="transparent")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.region_list = tk.Listbox(list_frame, bg="#1a1a2e", fg="#eaeaea",
                                      selectbackground="#e94560", selectforeground="#ffffff",
                                      font=("Consolas", 10), highlightthickness=0,
                                      activestyle="none", borderwidth=0)
        self.region_list.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(list_frame, command=self.region_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.region_list.config(yscrollcommand=scrollbar.set)
        self.region_list.bind("<<ListboxSelect>>", self._on_region_select)
        self.region_list.bind("<Delete>", self._on_region_delete)
        self.region_list.bind("<Double-Button-1>", self._on_region_double_click)

        # 按钮行
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))

        ctk.CTkButton(btn_frame, text="🗑 删除选中", width=100,
                       command=self.delete_region).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkButton(btn_frame, text="🧹 清空全部", width=100,
                       command=self.clear_regions).pack(side=tk.LEFT)

        # 编辑面板
        edit_frame = ctk.CTkScrollableFrame(tab, label_text="编辑选中区域",
                                             height=320)
        edit_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        edit_frame.grid_columnconfigure(1, weight=1)

        self.lbl_edit_info = ctk.CTkLabel(edit_frame, text="请选择一个水印区域进行编辑",
                                           text_color="gray50")
        self.lbl_edit_info.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # 类型
        ctk.CTkLabel(edit_frame, text="类型:").grid(row=1, column=0, sticky="w", pady=(0, 4))
        self.type_var = tk.StringVar(value="fixed")
        type_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        type_frame.grid(row=1, column=1, columnspan=2, sticky="w", pady=(0, 4))
        ctk.CTkRadioButton(type_frame, text="固定", variable=self.type_var,
                           value="fixed", command=self._on_type_changed).pack(side=tk.LEFT, padx=(0, 12))
        ctk.CTkRadioButton(type_frame, text="动态", variable=self.type_var,
                           value="dynamic", command=self._on_type_changed).pack(side=tk.LEFT)

        # 动态跟踪
        self.track_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(edit_frame, text="动态跟踪（帧差检测水印移动位置）",
                        variable=self.track_var, command=self._on_track_changed).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 8))

        # 时间段
        ctk.CTkLabel(edit_frame, text="时间段", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 2))

        ctk.CTkLabel(edit_frame, text="起:").grid(row=4, column=0, sticky="w")
        self.scale_start = ctk.CTkSlider(edit_frame, from_=0, to=100, command=self._on_start_changed)
        self.scale_start.grid(row=4, column=1, sticky="ew", padx=(4, 4))
        self.lbl_start_val = ctk.CTkLabel(edit_frame, text="0.0s", width=60)
        self.lbl_start_val.grid(row=4, column=2)

        ctk.CTkLabel(edit_frame, text="止:").grid(row=5, column=0, sticky="w")
        self.scale_end = ctk.CTkSlider(edit_frame, from_=0, to=100, command=self._on_end_changed)
        self.scale_end.grid(row=5, column=1, sticky="ew", padx=(4, 4))
        self.lbl_end_val = ctk.CTkLabel(edit_frame, text="0.0s", width=60)
        self.lbl_end_val.grid(row=5, column=2)

        jump_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        jump_frame.grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 4))
        ctk.CTkButton(jump_frame, text="跳到起点", width=80, command=self._jump_to_start).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkButton(jump_frame, text="跳到终点", width=80, command=self._jump_to_end).pack(side=tk.LEFT)

        # 位置/大小
        ctk.CTkLabel(edit_frame, text="位置 / 大小", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 2))

        for i, (label, attr) in enumerate([("X:", "x"), ("Y:", "y"), ("W:", "w"), ("H:", "h")]):
            row = 8 + i
            ctk.CTkLabel(edit_frame, text=label).grid(row=row, column=0, sticky="w")
            slider = ctk.CTkSlider(edit_frame, from_=0, to=100, command=self._on_pos_changed)
            slider.grid(row=row, column=1, sticky="ew", padx=(4, 4))
            val_label = ctk.CTkLabel(edit_frame, text="0", width=50)
            val_label.grid(row=row, column=2)
            setattr(self, f"scale_{attr}", slider)
            setattr(self, f"lbl_{attr}_val", val_label)

        self.scale_w.configure(from_=1)
        self.scale_h.configure(from_=1)

    def _build_method_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="选择修复算法", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 8))

        self.method_var = tk.StringVar(value="inpaint")
        row = 1
        for val, label, desc in self._METHODS:
            item_frame = ctk.CTkFrame(tab, fg_color="transparent")
            item_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
            item_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkRadioButton(item_frame, text=label, variable=self.method_var,
                               value=val, command=self._on_method_changed).pack(anchor="w")
            ctk.CTkLabel(item_frame, text=f"  {desc}", text_color="gray50",
                         font=ctk.CTkFont(size=11)).pack(anchor="w")
            row += 1

        # 修复半径
        radius_frame = ctk.CTkFrame(tab, fg_color="transparent")
        radius_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(12, 4))
        radius_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(radius_frame, text="修复半径:").grid(row=0, column=0, sticky="w")
        self._updating_sliders = True
        self.scale_radius = ctk.CTkSlider(radius_frame, from_=1, to=50, command=self._on_radius_changed)
        self.scale_radius.set(15)
        self.scale_radius.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self.lbl_radius_val = ctk.CTkLabel(radius_frame, text="15", width=40)
        self.lbl_radius_val.grid(row=0, column=2)
        self._updating_sliders = False
        row += 1

        # LaMa 质量
        self.lama_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.lama_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 8))
        self.lama_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.lama_frame, text="LaMa质量:").grid(row=0, column=0, sticky="w")
        self._updating_sliders = True
        self.lama_scale = ctk.CTkSlider(self.lama_frame, from_=0.25, to=1.0,
                                        command=self._on_lama_quality_changed)
        self.lama_scale.set(0.5)
        self.lama_scale.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self.lbl_lama_val = ctk.CTkLabel(self.lama_frame, text="0.5 (快速)", width=100)
        self.lbl_lama_val.grid(row=0, column=2)
        self._updating_sliders = False

    def _build_help_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        help_text = (
            "操作流程\n"
            "1. 点击「打开视频」加载视频文件\n"
            "2. 在画面上拖拽框选水印区域\n"
            "3. 在「水印区域」标签页调整区域参数\n"
            "4. 在「去水印方法」标签页选择算法\n"
            "5. 点击「预览效果」查看当前帧处理结果\n"
            "6. 满意后点击「开始处理」导出视频\n\n"
            "与剪映模糊去水印的区别\n"
            "• 剪映: 只能在水印上叠加模糊/马赛克，\n"
            "  水印痕迹依然可见，效果粗糙\n"
            "• Inpaint/NS: 从水印边缘向内推算修复，\n"
            "  能还原被遮挡的背景，痕迹更小\n"
            "• 贴片覆盖: 从外围取样填充+羽化，\n"
            "  适合复杂纹理背景\n"
            "• LaMa AI: 深度学习修复，能理解画面\n"
            "  语义，修复效果最自然，但速度较慢\n\n"
            "动态水印跟踪\n"
            "勾选「动态跟踪」后，程序会逐帧检测\n"
            "水印实际位置，自动追踪移动的水印"
        )
        help_label = ctk.CTkTextbox(tab, font=ctk.CTkFont(size=13), wrap="word")
        help_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        help_label.insert("1.0", help_text)
        help_label.configure(state="disabled")

    # ==================== 视频操作 ====================

    def open_video(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.webm"), ("所有文件", "*.*")]
        )
        if not path:
            return

        self.video_path = path
        self.lbl_video.configure(text=os.path.basename(path))

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 24

        if self.ffmpeg:
            self.video_info = get_video_info(self.ffmpeg, path)
        else:
            self.video_info = {"width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               "fps": fps, "duration": self.total_frames / fps}

        self.current_frame_idx = 0
        self.scale_timeline.configure(to=max(self.total_frames - 1, 0))
        self.scale_timeline.set(0)

        self.regions.clear()
        self._refresh_region_list()

        self._show_frame(0)
        self.lbl_status.configure(
            text=f"{self.video_info['width']}x{self.video_info['height']} | {fps:.1f}fps | {self.total_frames}帧")

    def _show_frame(self, idx):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.current_frame_idx = idx
            self.canvas.set_frame(frame)
            self._update_overlays()
            fps = self.video_info["fps"] if self.video_info else 24
            t = idx / fps
            self.lbl_frame_info.configure(text=f"{idx} / {self.total_frames - 1}  ({t:.1f}s)")
            if not self._seeking:
                self.scale_timeline.set(idx)

    def on_timeline_seek(self, val):
        idx = int(float(val))
        if idx != self.current_frame_idx:
            self._seeking = True
            self._show_frame(idx)
            self._seeking = False

    def prev_frame(self):
        self._show_frame(self.current_frame_idx - 1)

    def next_frame(self):
        self._show_frame(self.current_frame_idx + 1)

    # ==================== 区域管理 ====================

    def _on_region_selected(self, x, y, w, h):
        rtype = self.type_var.get()
        total = self.total_frames if self.total_frames else 1

        if rtype == "fixed":
            frame_start, frame_end = 0, total - 1
        else:
            frame_start = self.current_frame_idx
            frame_end = total - 1

        region = {
            "x": x, "y": y, "w": w, "h": h,
            "type": rtype,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "track": False,
        }
        self.regions.append(region)
        self._refresh_region_list()
        self._update_overlays()
        self.region_list.selection_set(len(self.regions) - 1)
        self.selected_region_idx = len(self.regions) - 1
        self._load_region_to_sliders(len(self.regions) - 1)

    def _refresh_region_list(self):
        self.region_list.delete(0, tk.END)
        for i, r in enumerate(self.regions):
            t_label = "固定" if r["type"] == "fixed" else "动态"
            track_label = " [跟踪]" if r.get("track") else ""
            entry = f'#{i+1} [{t_label}{track_label}] ({r["x"]},{r["y"]}) {r["w"]}x{r["h"]}'
            if r["type"] == "dynamic":
                fps = self.video_info["fps"] if self.video_info else 24
                entry += f' [{r["frame_start"]/fps:.1f}-{r["frame_end"]/fps:.1f}s]'
            self.region_list.insert(tk.END, entry)

    def _update_overlays(self):
        active = []
        for r in self.regions:
            if r["type"] == "fixed":
                active.append(r)
            elif r["frame_start"] <= self.current_frame_idx <= r["frame_end"]:
                active.append(r)
        self.canvas.set_regions(active)

    def _on_region_select(self, event):
        sel = self.region_list.curselection()
        if sel:
            self.selected_region_idx = sel[0]
            self._load_region_to_sliders(sel[0])

    def _on_region_delete(self, event):
        self.delete_region()

    def _on_region_double_click(self, event):
        sel = self.region_list.curselection()
        if not sel:
            return
        idx = sel[0]
        r = self.regions[idx]
        if r["type"] == "dynamic":
            self._show_frame(r["frame_start"])

    def delete_region(self):
        sel = self.region_list.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先选择一个水印区域")
            return
        idx = sel[0]
        del self.regions[idx]
        self.selected_region_idx = None
        self._refresh_region_list()
        self._update_overlays()
        self._clear_slider_panel()

    def clear_regions(self):
        if not self.regions:
            return
        if messagebox.askyesno("确认", "确定清空所有水印区域？"):
            self.regions.clear()
            self.selected_region_idx = None
            self._refresh_region_list()
            self._update_overlays()
            self._clear_slider_panel()

    # ---- 滑块编辑面板方法 ----

    def _clear_slider_panel(self):
        self.lbl_edit_info.configure(text="请选择一个水印区域进行编辑", text_color="gray50")
        self._updating_sliders = True
        self.track_var.set(False)
        self.scale_start.set(0)
        self.scale_end.set(0)
        self.scale_x.set(0)
        self.scale_y.set(0)
        self.scale_w.set(1)
        self.scale_h.set(1)
        self.lbl_start_val.configure(text="0.0s")
        self.lbl_end_val.configure(text="0.0s")
        self.lbl_x_val.configure(text="0")
        self.lbl_y_val.configure(text="0")
        self.lbl_w_val.configure(text="0")
        self.lbl_h_val.configure(text="0")
        self._updating_sliders = False

    def _load_region_to_sliders(self, idx):
        if idx is None or idx < 0 or idx >= len(self.regions):
            self._clear_slider_panel()
            return
        r = self.regions[idx]
        fps = self.video_info["fps"] if self.video_info else 24
        total = self.total_frames if self.total_frames else 1
        vw = self.video_info["width"] if self.video_info else r["x"] + r["w"]
        vh = self.video_info["height"] if self.video_info else r["y"] + r["h"]

        self._updating_sliders = True

        self.type_var.set(r["type"])
        self.track_var.set(r.get("track", False))
        self.lbl_edit_info.configure(text=f"编辑 #{idx+1}", text_color="gray80")

        self.scale_start.configure(to=max(total - 1, 0))
        self.scale_end.configure(to=max(total - 1, 0))
        self.scale_start.set(r["frame_start"])
        self.scale_end.set(r["frame_end"])
        self.lbl_start_val.configure(text=f"{r['frame_start']/fps:.1f}s")
        self.lbl_end_val.configure(text=f"{r['frame_end']/fps:.1f}s")

        self.scale_x.configure(to=max(vw, 1))
        self.scale_y.configure(to=max(vh, 1))
        self.scale_w.configure(to=max(vw, 1))
        self.scale_h.configure(to=max(vh, 1))
        self.scale_x.set(r["x"])
        self.scale_y.set(r["y"])
        self.scale_w.set(r["w"])
        self.scale_h.set(r["h"])
        self.lbl_x_val.configure(text=str(r["x"]))
        self.lbl_y_val.configure(text=str(r["y"]))
        self.lbl_w_val.configure(text=str(r["w"]))
        self.lbl_h_val.configure(text=str(r["h"]))

        self._updating_sliders = False

    def _get_selected_region(self):
        if self.selected_region_idx is None or self.selected_region_idx >= len(self.regions):
            return None
        return self.regions[self.selected_region_idx]

    def _on_type_changed(self):
        if self._updating_sliders:
            return
        r = self._get_selected_region()
        if r is None:
            return
        r["type"] = self.type_var.get()
        if r["type"] == "fixed":
            r["frame_start"] = 0
            r["frame_end"] = self.total_frames - 1 if self.total_frames else 0
        self._refresh_region_list()
        self._update_overlays()
        self._load_region_to_sliders(self.selected_region_idx)

    def _on_track_changed(self):
        if self._updating_sliders:
            return
        r = self._get_selected_region()
        if r is None:
            return
        r["track"] = self.track_var.get()
        self._refresh_region_list()

    def _on_start_changed(self, val):
        if self._updating_sliders:
            return
        r = self._get_selected_region()
        if r is None:
            return
        r["frame_start"] = int(float(val))
        fps = self.video_info["fps"] if self.video_info else 24
        self.lbl_start_val.configure(text=f"{r['frame_start']/fps:.1f}s")
        self._refresh_region_list()
        self._update_overlays()

    def _on_end_changed(self, val):
        if self._updating_sliders:
            return
        r = self._get_selected_region()
        if r is None:
            return
        r["frame_end"] = int(float(val))
        fps = self.video_info["fps"] if self.video_info else 24
        self.lbl_end_val.configure(text=f"{r['frame_end']/fps:.1f}s")
        self._refresh_region_list()
        self._update_overlays()

    def _on_pos_changed(self, val=None):
        if self._updating_sliders:
            return
        r = self._get_selected_region()
        if r is None:
            return
        r["x"] = int(self.scale_x.get())
        r["y"] = int(self.scale_y.get())
        r["w"] = max(1, int(self.scale_w.get()))
        r["h"] = max(1, int(self.scale_h.get()))
        self.lbl_x_val.configure(text=str(r["x"]))
        self.lbl_y_val.configure(text=str(r["y"]))
        self.lbl_w_val.configure(text=str(r["w"]))
        self.lbl_h_val.configure(text=str(r["h"]))
        self._refresh_region_list()
        self._update_overlays()

    def _on_radius_changed(self, val):
        if self._updating_sliders:
            return
        r = int(float(val))
        self.lbl_radius_val.configure(text=str(r))

    def _on_method_changed(self):
        if self.method_var.get() == "lama":
            self.lama_frame.grid()
        else:
            self.lama_frame.grid_forget()

    def _on_lama_quality_changed(self, val):
        if self._updating_sliders:
            return
        v = float(val)
        if v >= 0.9:
            label = "1.0 (原速)"
        elif v >= 0.6:
            label = f"{v:.2f} (高质量)"
        elif v >= 0.35:
            label = f"{v:.2f} (快速)"
        else:
            label = f"{v:.2f} (极速)"
        self.lbl_lama_val.configure(text=label)

    def _jump_to_start(self):
        r = self._get_selected_region()
        if r is None:
            return
        self._show_frame(r["frame_start"])

    def _jump_to_end(self):
        r = self._get_selected_region()
        if r is None:
            return
        self._show_frame(r["frame_end"])

    def _get_active_regions_for_frame(self, frame_idx):
        result = []
        for r in self.regions:
            if r["type"] == "fixed":
                result.append(r)
            elif r["frame_start"] <= frame_idx <= r["frame_end"]:
                result.append(r)
        return result

    def _detect_watermark_position(self, frame, region, prev_frame):
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        h, w = frame.shape[:2]

        rx = max(0, rx)
        ry = max(0, ry)
        rx2 = min(w, rx + rw)
        ry2 = min(h, ry + rh)

        if rx2 - rx < 5 or ry2 - ry < 5:
            return (rx, ry, rw, rh)

        cur_patch = frame[ry:ry2, rx:rx2]
        prev_patch = prev_frame[ry:ry2, rx:rx2]

        diff = cv2.absdiff(cur_patch, prev_patch)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (rx, ry, rw, rh)

        valid = [c for c in contours if cv2.contourArea(c) > 30]
        if not valid:
            return (rx, ry, rw, rh)

        all_points = np.vstack(valid)
        x, y, dw, dh = cv2.boundingRect(all_points)

        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        dw = min(rx2 - rx - x, dw + 2 * pad)
        dh = min(ry2 - ry - y, dh + 2 * pad)

        if dw < 5 or dh < 5:
            return (rx, ry, rw, rh)

        return (rx + x, ry + y, dw, dh)

    def _resolve_region_boxes(self, regions, frame, prev_frame):
        boxes = []
        for r in regions:
            if r.get("track") and prev_frame is not None:
                bx, by, bw, bh = self._detect_watermark_position(frame, r, prev_frame)
            else:
                bx, by, bw, bh = r["x"], r["y"], r["w"], r["h"]
            boxes.append((bx, by, bw, bh))
        return boxes

    # ==================== 帧处理 ====================

    def _process_frame(self, frame, regions, method, target_size, radius, mask_cache, prev_frame=None):
        if not regions:
            return frame

        boxes = self._resolve_region_boxes(regions, frame, prev_frame)

        if method == "crop":
            h, w = frame.shape[:2]
            top, bottom, left, right = 0, h, 0, w
            for (x, y, rw, rh) in boxes:
                top = max(top, y + rh)
                left = max(left, x + rw)
                bottom = min(bottom, y)
                right = min(right, x)
            top = min(top, h - 1)
            bottom = max(bottom, top + 1)
            left = min(left, w - 1)
            right = max(right, left + 1)
            cropped = frame[top:bottom, left:right]
            return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)

        elif method == "mosaic":
            result = frame.copy()
            h, w = frame.shape[:2]
            for (x, y, rw, rh) in boxes:
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + rw)
                y2 = min(h, y + rh)
                rw = x2 - x
                rh = y2 - y
                if rw <= 0 or rh <= 0:
                    continue
                block_size = max(4, radius)
                small = cv2.resize(frame[y:y2, x:x2],
                                   (max(1, rw // block_size), max(1, rh // block_size)),
                                   interpolation=cv2.INTER_AREA)
                mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                result[y:y2, x:x2] = mosaic
            return result

        elif method == "blur":
            result = frame.copy()
            h, w = frame.shape[:2]
            for (x, y, rw, rh) in boxes:
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + rw)
                y2 = min(h, y + rh)
                rw = x2 - x
                rh = y2 - y
                if rw <= 0 or rh <= 0:
                    continue
                ksize = max(3, radius * 2 + 1)
                ksize = min(ksize, rw, rh)
                if ksize % 2 == 0:
                    ksize -= 1
                if ksize >= 3:
                    blurred = cv2.GaussianBlur(frame[y:y2, x:x2], (ksize, ksize), 0)
                    result[y:y2, x:x2] = blurred
            return result

        elif method == "block":
            result = frame.copy()
            h, w = frame.shape[:2]
            for (x, y, rw, rh) in boxes:
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + rw)
                y2 = min(h, y + rh)
                if x2 > x and y2 > y:
                    result[y:y2, x:x2] = (0, 0, 0)
            return result

        elif method == "cover":
            result = frame.copy()
            h, w = frame.shape[:2]
            for (x, y, rw, rh) in boxes:
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + rw)
                y2 = min(h, y + rh)
                rw = x2 - x
                rh = y2 - y
                if rw <= 0 or rh <= 0:
                    continue

                margin = max(rw, rh, 10)
                src_x1 = max(0, x - margin)
                src_y1 = max(0, y - margin)
                src_x2 = min(w, x2 + margin)
                src_y2 = min(h, y2 + margin)

                src_region = frame[src_y1:src_y2, src_x1:src_x2].copy()
                src_mask = np.ones((src_y2 - src_y1, src_x2 - src_x1), dtype=np.uint8) * 255
                wm_x1 = x - src_x1
                wm_y1 = y - src_y1
                wm_x2 = wm_x1 + rw
                wm_y2 = wm_y1 + rh
                src_mask[max(0, wm_y1):min(src_mask.shape[0], wm_y2),
                         max(0, wm_x1):min(src_mask.shape[1], wm_x2)] = 0

                patch = cv2.inpaint(src_region, cv2.bitwise_not(src_mask),
                                    max(rw, rh, 15), cv2.INPAINT_TELEA)

                filled = patch[max(0, wm_y1):min(patch.shape[0], wm_y2),
                               max(0, wm_x1):min(patch.shape[1], wm_x2)]

                feather = min(8, rw // 2, rh // 2)
                if feather > 0:
                    blend_mask = np.zeros((rh, rw), dtype=np.float32)
                    blend_mask[feather:-feather, feather:-feather] = 1.0
                    blend_mask = cv2.GaussianBlur(blend_mask, (feather * 2 + 1, feather * 2 + 1), 0)
                    blend_mask = np.clip(blend_mask, 0, 1)
                    blend_mask_3ch = cv2.merge([blend_mask, blend_mask, blend_mask])
                    target_area = result[y:y2, x:x2].astype(np.float32)
                    filled_area = filled.astype(np.float32)
                    blended = (filled_area * blend_mask_3ch +
                               target_area * (1 - blend_mask_3ch)).astype(np.uint8)
                    result[y:y2, x:x2] = blended
                else:
                    result[y:y2, x:x2] = filled

            return result

        elif method in ("inpaint", "ns"):
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for (x, y, rw, rh) in boxes:
                x2 = min(w, x + rw)
                y2 = min(h, y + rh)
                mask[max(0, y):y2, max(0, x):x2] = 255
            flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
            return cv2.inpaint(frame, mask, radius, flag)

        elif method == "lama":
            result = frame.copy()
            h, w = frame.shape[:2]
            lama = get_lama()
            scale = float(self.lama_scale.get())
            small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            small_result = small.copy()

            for (x, y, rw, rh) in boxes:
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + rw)
                y2 = min(h, y + rh)
                rw = x2 - x
                rh = y2 - y
                if rw <= 0 or rh <= 0:
                    continue

                margin = max(rw, rh, 20)
                px1 = max(0, int((x - margin) * scale))
                py1 = max(0, int((y - margin) * scale))
                px2 = min(sw, int((x2 + margin) * scale))
                py2 = min(sh, int((y2 + margin) * scale))

                patch = small[py1:py2, px1:px2].copy()
                patch_mask = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
                mx1 = max(0, int(x * scale) - px1)
                my1 = max(0, int(y * scale) - py1)
                mx2 = min(px2 - px1, int(x2 * scale) - px1)
                my2 = min(py2 - py1, int(y2 * scale) - py1)
                patch_mask[my1:my2, mx1:mx2] = 255

                rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_patch)
                pil_mask = Image.fromarray(patch_mask)
                result_pil = lama(pil_img, pil_mask)
                result_patch = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                if result_patch.shape[:2] != (py2 - py1, px2 - px1):
                    result_patch = cv2.resize(result_patch, (px2 - px1, py2 - py1), interpolation=cv2.INTER_LANCZOS4)
                small_result[py1:py2, px1:px2] = result_patch

            result = cv2.resize(small_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
            return result

        return frame

    # ==================== 预览 & 处理 ====================

    def preview_result(self):
        if self.current_frame is None:
            messagebox.showwarning("提示", "请先打开视频")
            return
        if not self.regions:
            messagebox.showwarning("提示", "请先标注水印区域")
            return

        method = self.method_var.get()
        radius = int(self.scale_radius.get())
        w, h = self.video_info["width"], self.video_info["height"]
        regions = self._get_active_regions_for_frame(self.current_frame_idx)

        if not regions:
            messagebox.showinfo("提示", "当前帧没有活跃的水印区域")
            return

        prev_frame = None
        if any(r.get("track") for r in regions) and self.current_frame_idx > 0:
            saved_idx = self.current_frame_idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx - 1)
            ret, prev_frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, saved_idx)
            if ret:
                self.cap.read()

        result = self._process_frame(self.current_frame.copy(), regions, method, (w, h), radius, {}, prev_frame)

        preview_win = ctk.CTkToplevel(self)
        preview_win.title("预览效果")
        preview_win.geometry("1200x400")

        orig_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        combined = np.hstack([orig_rgb, result_rgb])
        pil_img = Image.fromarray(combined)
        scale = min(1800 / pil_img.width, 600 / pil_img.height, 1.0)
        pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        lbl = tk.Label(preview_win, image=photo, bg="#0a0a0f")
        lbl.image = photo
        lbl.pack(padx=10, pady=10)

        ctk.CTkLabel(preview_win, text=f"方法: {method} | 区域数: {len(regions)} | 左:原图  右:处理后").pack(pady=(0, 10))

    def start_process(self):
        if not self.video_path:
            messagebox.showwarning("提示", "请先打开视频")
            return
        if not self.regions:
            messagebox.showwarning("提示", "请先标注水印区域")
            return
        if self.processing:
            return

        output_path = filedialog.asksaveasfilename(
            title="保存输出视频",
            defaultextension=".mp4",
            initialfile="output_" + os.path.basename(self.video_path),
            filetypes=[("MP4 视频", "*.mp4"), ("所有文件", "*.*")]
        )
        if not output_path:
            return

        self.processing = True
        self.btn_process.configure(state="disabled")
        self.btn_preview.configure(state="disabled")

        method = self.method_var.get()
        radius = int(self.scale_radius.get())

        thread = threading.Thread(target=self._process_video, args=(output_path, method, radius), daemon=True)
        thread.start()

    def _process_video(self, output_path, method, radius):
        try:
            w, h = self.video_info["width"], self.video_info["height"]
            fps = self.video_info["fps"]

            temp_path = output_path + ".tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

            if not out.isOpened():
                self.after(0, lambda: messagebox.showerror("错误", "无法创建输出视频"))
                return

            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.after(0, lambda: self.progress.configure(mode="determinate"))
            self.after(0, lambda: self.progress.configure(determinate_speed=1))
            self.after(0, lambda: self.progress.set(0))

            mask_cache = {}
            prev_frame = None
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                regions = self._get_active_regions_for_frame(idx)
                processed = self._process_frame(frame, regions, method, (w, h), radius, mask_cache, prev_frame)
                out.write(processed)
                prev_frame = frame
                idx += 1

                if idx % 10 == 0:
                    self.after(0, lambda v=idx, t=total: self._update_progress(v, t))

            cap.release()
            out.release()

            self.after(0, lambda: self.lbl_progress.configure(text="合并音频..."))

            if self.ffmpeg:
                final_path = output_path + ".final.mp4"
                cmd = [
                    self.ffmpeg,
                    "-i", temp_path,
                    "-i", self.video_path,
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-y",
                    final_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if result.returncode == 0:
                    os.replace(final_path, output_path)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                else:
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    os.replace(temp_path, output_path)
            else:
                os.replace(temp_path, output_path)

            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self.lbl_progress.configure(text="完成!"))
            self.after(0, lambda: messagebox.showinfo("完成", f"视频已保存到:\n{output_path}"))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.processing = False
            self.after(0, lambda: self.btn_process.configure(state="normal"))
            self.after(0, lambda: self.btn_preview.configure(state="normal"))

    def _update_progress(self, current, total):
        if total > 0:
            self.progress.set(current / total)
        self.lbl_progress.configure(text=f"{current}/{total}")


def main():
    app = WatermarkRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
