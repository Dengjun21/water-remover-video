"""
豆包AI视频去水印脚本
支持三种去水印方式：
  - crop: 裁剪掉水印所在边缘，再缩放回原尺寸（最彻底，推荐）
  - inpaint: OpenCV TELEA 算法逐帧修复
  - ns: OpenCV Navier-Stokes 算法逐帧修复

用法:
  python remove_watermark.py input.mp4 output.mp4
  python remove_watermark.py input.mp4 output.mp4 --method crop
  python remove_watermark.py input.mp4 output.mp4 --tl 0,0,200,50 --br 1030,665,250,55
  python remove_watermark.py input.mp4 output.mp4 --method inpaint --radius 8
"""

import argparse
import os
import sys
import shutil
import subprocess

import cv2
import numpy as np

_lama_instance = None


def get_lama():
    global _lama_instance
    if _lama_instance is None:
        try:
            from simple_lama_inpainting import SimpleLama
            _lama_instance = SimpleLama()
        except ImportError:
            print("错误: LaMa 模型未安装，请运行: pip install simple-lama-inpainting")
            sys.exit(1)
    return _lama_instance


def find_ffmpeg():
    """查找系统 ffmpeg，找不到则回退到 imageio_ffmpeg"""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("错误: 未找到 ffmpeg，请安装 ffmpeg 或 pip install imageio-ffmpeg")
        sys.exit(1)


def get_video_info(ffmpeg, video_path):
    """获取视频分辨率、帧率、时长"""
    cmd = [ffmpeg, "-i", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    info = {"width": 0, "height": 0, "fps": 0, "duration": 0}
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


def create_mask(width, height, regions):
    """
    根据水印区域列表创建掩码图像。
    regions: [(x, y, w, h), ...]
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for (x, y, w, h) in regions:
        x = max(0, x)
        y = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        mask[y:y2, x:x2] = 255
    return mask


def process_frame_inpaint(frame, mask, radius=5):
    """使用 OpenCV inpaint (TELEA) 修复单帧"""
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)


def process_frame_cover(frame, regions, radius=15):
    """
    贴片覆盖：从水印外围取像素，用 OpenCV inpaint 填充水印区域，边缘羽化混合。
    regions: [(x, y, w, h), ...]
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    for (x, y, rw, rh) in regions:
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


def process_frame_ns(frame, mask, radius=5):
    """使用 OpenCV inpaint (Navier-Stokes) 修复单帧"""
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_NS)


def process_frame_blur(frame, mask, blur_size=25):
    """使用模糊覆盖方式去除水印"""
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    # 将 mask 扩展以使边缘更自然
    kernel = np.ones((blur_size, blur_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    expanded_mask = cv2.GaussianBlur(expanded_mask, (blur_size, blur_size), 0)

    mask_3ch = cv2.merge([expanded_mask, expanded_mask, expanded_mask])
    mask_float = mask_3ch.astype(np.float32) / 255.0
    result = (frame.astype(np.float32) * (1 - mask_float) +
              blurred.astype(np.float32) * mask_float)
    return result.astype(np.uint8)


def process_frame_crop(frame, tl, br, target_size):
    """
    裁剪掉水印所在的边缘区域，然后缩放回原始尺寸。
    tl: (x, y, w, h) 左上角水印区域 -> 裁掉 top=y+h, left=x+w
    br: (x, y, w, h) 右下角水印区域 -> 裁掉 bottom=y, right=x
    target_size: (width, height) 原始尺寸，用于缩放回去
    """
    h, w = frame.shape[:2]
    top = 0
    bottom = h
    left = 0
    right = w

    if tl:
        tx, ty, tw, th = tl
        top = max(top, ty + th)
        left = max(left, tx + tw)
    if br:
        bx, by, bw, bh = br
        bottom = min(bottom, by)
        right = min(right, bx)

    # 确保裁剪区域有效
    top = min(top, h - 1)
    bottom = max(bottom, top + 1)
    left = min(left, w - 1)
    right = max(right, left + 1)

    cropped = frame[top:bottom, left:right]
    # 缩放回原始尺寸
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    return resized


def remove_watermark(input_path, output_path, tl=None, br=None, method="crop",
                     inpaint_radius=5, blur_size=25, ffmpeg=None):
    """
    去除视频水印。

    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        tl: (x, y, w, h) 左上角水印区域
        br: (x, y, w, h) 右下角水印区域
        method: crop(裁剪缩放) / inpaint(TELEA修复) / ns(Navier-Stokes修复)
        inpaint_radius: inpaint 修复半径
        blur_size: 模糊核大小
        ffmpeg: ffmpeg 路径
    """
    if ffmpeg is None:
        ffmpeg = find_ffmpeg()

    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    info = get_video_info(ffmpeg, input_path)
    w, h = info["width"], info["height"]
    fps = info["fps"] or 24
    print(f"视频: {w}x{h}, {fps}fps, {info['duration']:.1f}s")

    # 默认水印区域（基于 1280x720 豆包AI视频）
    if tl is None:
        tl = (0, 0, min(200, w), min(50, h))
    if br is None:
        br = (max(0, w - 250), max(0, h - 55), min(250, w), min(55, h))

    if tl:
        print(f"左上角水印区域: x={tl[0]}, y={tl[1]}, w={tl[2]}, h={tl[3]}")
    if br:
        print(f"右下角水印区域: x={br[0]}, y={br[1]}, w={br[2]}, h={br[3]}")

    print(f"去水印方法: {method}")

    if method == "crop":
        # 计算裁剪后保留的区域
        crop_top = tl[1] + tl[3] if tl else 0
        crop_bottom = br[1] if br else h
        crop_left = tl[0] + tl[2] if tl else 0
        crop_right = br[0] if br else w
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        print(f"裁剪区域: ({crop_left},{crop_top})-({crop_right},{crop_bottom}) = {crop_w}x{crop_h}")
        print(f"缩放回: {w}x{h}")

    # 创建掩码（inpaint/ns/lama 方法用）
    all_regions = []
    if tl:
        all_regions.append(tl)
    if br:
        all_regions.append(br)

    mask = None
    if method in ("inpaint", "ns", "lama"):
        mask = create_mask(w, h, all_regions)
        regions = all_regions

    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print("错误: 无法创建输出视频")
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if method == "crop":
            processed = process_frame_crop(frame, tl, br, (w, h))
        elif method == "cover":
            processed = process_frame_cover(frame, all_regions, inpaint_radius)
        elif method == "ns":
            processed = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_NS)
        elif method == "lama":
            from PIL import Image
            lama = get_lama()
            h, w = frame.shape[:2]
            scale = 0.5
            small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            small_result = small.copy()
            for (mx, my, mw, mh) in all_regions:
                mx = max(0, mx)
                my = max(0, my)
                mx2 = min(w, mx + mw)
                my2 = min(h, my + mh)
                mw = mx2 - mx
                mh = my2 - my
                if mw <= 0 or mh <= 0:
                    continue
                margin = max(mw, mh, 20)
                px1 = max(0, int((mx - margin) * scale))
                py1 = max(0, int((my - margin) * scale))
                px2 = min(sw, int((mx2 + margin) * scale))
                py2 = min(sh, int((my2 + margin) * scale))
                patch = small[py1:py2, px1:px2].copy()
                patch_mask = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
                qx1 = max(0, int(mx * scale) - px1)
                qy1 = max(0, int(my * scale) - py1)
                qx2 = min(px2 - px1, int(mx2 * scale) - px1)
                qy2 = min(py2 - py1, int(my2 * scale) - py1)
                patch_mask[qy1:qy2, qx1:qx2] = 255
                rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_patch)
                pil_mask = Image.fromarray(patch_mask)
                result_pil = lama(pil_img, pil_mask)
                result_patch = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                if result_patch.shape[:2] != (py2 - py1, px2 - px1):
                    result_patch = cv2.resize(result_patch, (px2 - px1, py2 - py1), interpolation=cv2.INTER_LANCZOS4)
                small_result[py1:py2, px1:px2] = result_patch
            processed = cv2.resize(small_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            processed = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

        out.write(processed)
        frame_idx += 1

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            pct = 100 * frame_idx / max(total_frames, 1)
            sys.stdout.write(f"\r进度: {frame_idx}/{total_frames} ({pct:.0f}%)")
            sys.stdout.flush()

    print()
    cap.release()
    out.release()

    # 用 ffmpeg 重封装为标准 mp4（修复音频 + 容器兼容性）
    temp_path = output_path + ".tmp.mp4"
    print("重封装视频（合并音频）...")
    cmd = [
        ffmpeg,
        "-i", output_path,
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-y",
        temp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        os.replace(temp_path, output_path)
    else:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"警告: 音频合并失败，输出视频无音频")

    print(f"完成! 输出文件: {output_path}")


def parse_box(s):
    """解析 'x,y,w,h' 格式的水印区域参数"""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("格式应为 x,y,w,h")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="豆包AI视频去水印工具 (OpenCV inpaint)")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", help="输出视频文件路径")
    parser.add_argument("--tl", type=parse_box, default=None,
                        help="左上角水印区域，格式: x,y,w,h (默认: 0,0,200,50)")
    parser.add_argument("--br", type=parse_box, default=None,
                        help="右下角水印区域，格式: x,y,w,h (默认: 1030,665,250,55)")
    parser.add_argument("--method", choices=["crop", "cover", "inpaint", "ns", "lama"], default="cover",
                        help="去水印方法: cover(贴片覆盖,推荐), crop(裁剪缩放), inpaint(TELEA修复), ns(Navier-Stokes修复), lama(LaMa AI修复)")
    parser.add_argument("--radius", type=int, default=5,
                        help="inpaint 修复半径 (默认: 5)")
    args = parser.parse_args()

    remove_watermark(
        input_path=args.input,
        output_path=args.output,
        tl=args.tl,
        br=args.br,
        method=args.method,
        inpaint_radius=args.radius,
    )


if __name__ == "__main__":
    main()
