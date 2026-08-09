import json
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from av.codec import CodecContext
from av.error import InvalidDataError

import cv2
import numpy as np
import scrcpy

GRID_SIZE = 24
BASE_TEMPLATE_WIDTH = 1920
BASE_TEMPLATE_HEIGHT = 1080
DRAW_PALETTE_INDICES = (0, 1, 3)
BACKGROUND_PALETTE_INDEX = 3
VALIDATION_MAX_ATTEMPTS = 5
DEFAULT_PALETTE_NAMES = (
    "black",
    "light_gray",
    "beige",
    "white",
)


logger = logging.getLogger("bad_paint_arknights")


@dataclass(frozen=True)
class PaletteColor:
    index: int
    name: str
    center: tuple[int, int]
    bgr: tuple[int, int, int]


@dataclass(frozen=True)
class TouchAction:
    kind: str
    start: tuple[int, int]
    end: tuple[int, int]


class SafeScrcpyClient(scrcpy.Client):
    def _Client__stream_loop(self) -> None:
        codec = CodecContext.create("h264", "r")
        while self.alive:
            try:
                raw_h264 = self._Client__video_socket.recv(0x10000)
                if not raw_h264:
                    if self.alive:
                        time.sleep(0.01)
                    continue

                packets = codec.parse(raw_h264)
                for packet in packets:
                    try:
                        frames = codec.decode(packet)
                    except InvalidDataError:
                        if self.alive:
                            logger.debug("Discarded malformed scrcpy video packet")
                        continue

                    for frame in frames:
                        frame = frame.to_ndarray(format="bgr24")
                        if self.flip:
                            frame = cv2.flip(frame, 1)
                        self.last_frame = frame
                        self.resolution = (frame.shape[1], frame.shape[0])
                        self._Client__send_to_listeners(scrcpy.EVENT_FRAME, frame)
            except BlockingIOError:
                time.sleep(0.01)
                if not self.block_frame:
                    self._Client__send_to_listeners(scrcpy.EVENT_FRAME, None)
            except OSError:
                if self.alive:
                    raise


@dataclass
class Config:
    project_root: Path
    assets_dir: Path
    templates_dir: Path
    screenshots_dir: Path
    debug_dir: Path
    output_video_path: Path

    adb_path: str = "adb"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    emulator_serial: str | None = None

    draw_fps: float = 5.0
    frame_start: int | None = None
    frame_end: int | None = None
    frame_threshold: int = 127
    action_delay_sec: float = 0.08

    template_score_threshold: float = 0.55
    clear_confirm_timeout_sec: float = 3.0
    frame_wait_timeout_sec: float = 5.0
    log_level: str = "INFO"


def load_config_from_json(project_root: Path) -> Config:
    config = Config(
        project_root=project_root,
        assets_dir=project_root / "assets",
        templates_dir=project_root / "assets" / "templates",
        screenshots_dir=project_root / "screenshots",
        debug_dir=project_root / "debug",
        output_video_path=project_root / "output.mp4",
    )

    config_path = project_root / "config.json"
    if not config_path.exists():
        logger.info("config.json not found, using built-in defaults")
        return config

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def _path_from_value(value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return project_root / p

    if "assets_dir" in data:
        config.assets_dir = _path_from_value(data["assets_dir"])
    if "templates_dir" in data:
        config.templates_dir = _path_from_value(data["templates_dir"])
    if "screenshots_dir" in data:
        config.screenshots_dir = _path_from_value(data["screenshots_dir"])
    if "output_video_path" in data:
        config.output_video_path = _path_from_value(data["output_video_path"])
    if "debug_dir" in data:
        config.debug_dir = _path_from_value(data["debug_dir"])

    if "adb_path" in data:
        config.adb_path = str(data["adb_path"])
    if "ffmpeg_path" in data:
        config.ffmpeg_path = str(data["ffmpeg_path"])
    if "ffprobe_path" in data:
        config.ffprobe_path = str(data["ffprobe_path"])
    if "emulator_serial" in data:
        config.emulator_serial = data["emulator_serial"]

    if "draw_fps" in data:
        config.draw_fps = float(data["draw_fps"])
    if "frame_start" in data:
        config.frame_start = None if data["frame_start"] is None else int(data["frame_start"])
    if "frame_end" in data:
        config.frame_end = None if data["frame_end"] is None else int(data["frame_end"])
    if "frame_threshold" in data:
        config.frame_threshold = int(data["frame_threshold"])
    if "action_delay_sec" in data:
        config.action_delay_sec = float(data["action_delay_sec"])
    if "template_score_threshold" in data:
        config.template_score_threshold = float(data["template_score_threshold"])
    if "clear_confirm_timeout_sec" in data:
        config.clear_confirm_timeout_sec = float(data["clear_confirm_timeout_sec"])
    if "frame_wait_timeout_sec" in data:
        config.frame_wait_timeout_sec = float(data["frame_wait_timeout_sec"])
    if "log_level" in data:
        config.log_level = str(data["log_level"])

    logger.info("Loaded configuration from %s", config_path)
    return config


def save_debug_screenshot(debug_dir: Path, frame: np.ndarray, reason: str) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason)
    path = debug_dir / f"{timestamp}_{safe_reason}.png"

    ok = cv2.imwrite(str(path), frame)
    if ok:
        logger.warning("Saved debug screenshot: %s", path)
    else:
        logger.warning("Failed to save debug screenshot: %s", path)

    return path


def save_base_debug_image(
    debug_dir: Path,
    frame: np.ndarray,
    canvas_rect: tuple[int, int, int, int],
    palette_rect: tuple[int, int, int, int],
    clear_button_rect: tuple[int, int, int, int],
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "base.png"

    debug_frame = frame.copy()

    def _draw_rect(
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        x, y, w, h = rect
        cv2.rectangle(debug_frame, (x, y), (x + w - 1, y + h - 1), color, 3)
        cv2.putText(
            debug_frame,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_canvas_grid(rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        for col in range(1, GRID_SIZE):
            gx = round(x + col * w / GRID_SIZE)
            cv2.line(debug_frame, (gx, y), (gx, y + h - 1), (0, 200, 255), 1, cv2.LINE_AA)

        for row in range(1, GRID_SIZE):
            gy = round(y + row * h / GRID_SIZE)
            cv2.line(debug_frame, (x, gy), (x + w - 1, gy), (0, 200, 255), 1, cv2.LINE_AA)

    _draw_rect(canvas_rect, (0, 255, 0), "canvas")
    _draw_canvas_grid(canvas_rect)
    _draw_rect(palette_rect, (255, 0, 0), "palette")
    _draw_rect(clear_button_rect, (0, 0, 255), "clear")

    ok = cv2.imwrite(str(path), debug_frame)
    if not ok:
        raise RuntimeError(f"Failed to write base debug image: {path}")

    logger.info("Saved base debug image: %s", path)
    return path


class ScrcpySession:
    def __init__(self, serial: str | None, max_fps: int):
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        self.client = SafeScrcpyClient(device=serial, max_fps=max_fps, block_frame=False)
        self.client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)

    def _on_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def start(self) -> None:
        self.client.start(threaded=True)

    def stop(self) -> None:
        self.client.stop()

    def get_frame(self, timeout_sec: float) -> np.ndarray:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            time.sleep(0.01)
        raise RuntimeError("Timed out waiting for scrcpy video frame")


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ensure_adb_connection(config: Config) -> None:
    logger.info("Checking ADB connection")
    if config.emulator_serial:
        logger.info("Connecting to emulator via ADB: %s", config.emulator_serial)
        connect_result = run_command([config.adb_path, "connect", config.emulator_serial])
        if connect_result.returncode != 0:
            raise RuntimeError(f"ADB connect failed: {connect_result.stderr.strip()}")

    devices_result = run_command([config.adb_path, "devices"])
    if devices_result.returncode != 0:
        raise RuntimeError(f"ADB devices failed: {devices_result.stderr.strip()}")

    lines = [line.strip() for line in devices_result.stdout.splitlines() if line.strip()]
    online = [line.split()[0] for line in lines[1:] if "\tdevice" in line]

    if config.emulator_serial:
        if config.emulator_serial not in online:
            raise RuntimeError(
                f"ADB device {config.emulator_serial} is not online. Found devices: {online}"
            )
    elif not online:
        raise RuntimeError("No ADB devices are online")

    logger.info("ADB is ready. Online devices: %s", ", ".join(online))


def get_first_video_file(assets_dir: Path) -> Path:
    extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")
    video_files: list[Path] = []
    for ext in extensions:
        video_files.extend(sorted(assets_dir.glob(ext)))

    if not video_files:
        raise RuntimeError("No compatible video file found in assets directory")
    logger.info("Selected source video: %s", video_files[0])
    return video_files[0]


def clear_screenshots_dir(path: Path) -> None:
    logger.info("Clearing screenshots directory: %s", path)
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def clear_debug_frames_dir(debug_dir: Path) -> None:
    frames_dir = debug_dir / "frames"
    logger.info("Clearing debug frames directory: %s", frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for item in frames_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def match_template_multiscale(
    image: np.ndarray,
    template: np.ndarray,
    score_threshold: float,
) -> tuple[int, int, int, int, float]:
    img_h, img_w = image.shape[:2]
    base_scale = min(img_w / BASE_TEMPLATE_WIDTH, img_h / BASE_TEMPLATE_HEIGHT)
    scales = np.linspace(base_scale * 0.6, base_scale * 1.4, 17)

    best = (-1, -1, 0, 0, -1.0)

    for scale in scales:
        if scale <= 0:
            continue

        tw = max(1, int(template.shape[1] * scale))
        th = max(1, int(template.shape[0] * scale))
        if tw >= img_w or th >= img_h:
            continue

        resized_template = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best[4]:
            best = (max_loc[0], max_loc[1], tw, th, float(max_val))

    if best[4] < score_threshold:
        raise RuntimeError(
            f"Template matching failed. score={best[4]:.3f}, threshold={score_threshold:.3f}"
        )

    logger.debug(
        "Template matched at x=%d y=%d w=%d h=%d score=%.3f",
        best[0],
        best[1],
        best[2],
        best[3],
        best[4],
    )

    return best


def detect_canvas_rect(screen: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    screen_h, screen_w = screen.shape[:2]
    min_side = int(0.5 * screen_h)
    min_rect_area = 0.18 * float(screen_h * screen_w)

    best_score = -1.0
    best_rect: tuple[int, int, int, int] | None = None
    best_threshold: int | None = None

    for threshold in (245, 235, 225, 215):
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Keep contour edges coherent without aggressively eroding thin borders.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h_rect = cv2.boundingRect(contour)
            side = min(w, h_rect)
            if side < min_side:
                continue

            ratio = w / max(h_rect, 1)
            if ratio < 0.75 or ratio > 1.35:
                continue

            rect_area = float(w * h_rect)
            if rect_area < min_rect_area:
                continue

            contour_area = float(cv2.contourArea(contour))
            fill_ratio = contour_area / max(rect_area, 1.0)
            if fill_ratio < 0.70:
                continue

            cx = x + w * 0.5
            cy = y + h_rect * 0.5
            norm_dx = abs(cx - screen_w * 0.5) / max(screen_w * 0.5, 1.0)
            norm_dy = abs(cy - screen_h * 0.5) / max(screen_h * 0.5, 1.0)

            square_score = 1.0 - min(abs(1.0 - ratio), 0.5)
            center_score = 1.0 - min(norm_dx + norm_dy, 1.5) / 1.5
            score = rect_area * (0.7 * square_score + 0.3 * center_score)

            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h_rect)
                best_threshold = threshold

    if best_rect is None:
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            x, y, w, h_rect = cv2.boundingRect(approx)
            side = min(w, h_rect)
            if side < min_side:
                continue

            rect_area = float(w * h_rect)
            if rect_area < min_rect_area:
                continue

            ratio = w / max(h_rect, 1)
            if ratio < 0.75 or ratio > 1.35:
                continue

            cx = x + w * 0.5
            cy = y + h_rect * 0.5
            norm_dx = abs(cx - screen_w * 0.5) / max(screen_w * 0.5, 1.0)
            norm_dy = abs(cy - screen_h * 0.5) / max(screen_h * 0.5, 1.0)
            square_score = 1.0 - min(abs(1.0 - ratio), 0.5)
            center_score = 1.0 - min(norm_dx + norm_dy, 1.5) / 1.5
            score = rect_area * (0.75 * square_score + 0.25 * center_score)

            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h_rect)
                best_threshold = -1

    if best_rect is None:
        raise RuntimeError("Failed to detect canvas: no square-like bright region found")

    def _refine_canvas_rect_with_dark_border(
        image_gray: np.ndarray,
        coarse_rect: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x, y, w, h_rect = coarse_rect
        search_margin = max(6, int(min(w, h_rect) * 0.05))

        strip_trim_y = max(4, int(h_rect * 0.08))
        strip_trim_x = max(4, int(w * 0.08))

        y0 = max(0, y + strip_trim_y)
        y1 = min(screen_h, y + h_rect - strip_trim_y)
        x0 = max(0, x + strip_trim_x)
        x1 = min(screen_w, x + w - strip_trim_x)
        if y1 <= y0 or x1 <= x0:
            return coarse_rect

        def _darkest_vertical(x_start: int, x_end: int, prefer_leftmost: bool) -> int | None:
            x_start = max(0, x_start)
            x_end = min(screen_w - 1, x_end)
            if x_end < x_start:
                return None

            best_x: int | None = None
            best_val = float("inf")
            for cx in range(x_start, x_end + 1):
                val = float(np.mean(image_gray[y0:y1, cx]))
                if val < best_val:
                    best_val = val
                    best_x = cx
                elif val == best_val and best_x is not None:
                    if prefer_leftmost and cx < best_x:
                        best_x = cx
                    if not prefer_leftmost and cx > best_x:
                        best_x = cx
            return best_x

        def _darkest_horizontal(y_start: int, y_end: int, prefer_topmost: bool) -> int | None:
            y_start = max(0, y_start)
            y_end = min(screen_h - 1, y_end)
            if y_end < y_start:
                return None

            best_y: int | None = None
            best_val = float("inf")
            for cy in range(y_start, y_end + 1):
                val = float(np.mean(image_gray[cy, x0:x1]))
                if val < best_val:
                    best_val = val
                    best_y = cy
                elif val == best_val and best_y is not None:
                    if prefer_topmost and cy < best_y:
                        best_y = cy
                    if not prefer_topmost and cy > best_y:
                        best_y = cy
            return best_y

        # Outward-only snap windows: do not shrink the coarse rect.
        left = _darkest_vertical(x - search_margin, x + 2, prefer_leftmost=True)
        right = _darkest_vertical(x + w - 3, x + w - 1 + search_margin, prefer_leftmost=False)
        top = _darkest_horizontal(y - search_margin, y + 2, prefer_topmost=True)
        bottom = _darkest_horizontal(y + h_rect - 3, y + h_rect - 1 + search_margin, prefer_topmost=False)

        if left is None or right is None or top is None or bottom is None:
            return coarse_rect

        left = min(left, x)
        right = max(right, x + w - 1)
        top = min(top, y)
        bottom = max(bottom, y + h_rect - 1)

        refined_w = right - left + 1
        refined_h = bottom - top + 1
        if refined_w <= 0 or refined_h <= 0:
            return coarse_rect

        return (left, top, refined_w, refined_h)

    refined_rect = _refine_canvas_rect_with_dark_border(gray, best_rect)
    if refined_rect != best_rect:
        logger.info(
            "Canvas refined by border snap: coarse=(%d,%d,%d,%d) refined=(%d,%d,%d,%d)",
            best_rect[0],
            best_rect[1],
            best_rect[2],
            best_rect[3],
            refined_rect[0],
            refined_rect[1],
            refined_rect[2],
            refined_rect[3],
        )
        best_rect = refined_rect

    # the inset is recorded from a 1080p screenshot
    inset_px = max(1, round(screen_h * (4.0 / 1080.0)))
    bx, by, bw, bh = best_rect
    if bw > inset_px * 2 and bh > inset_px * 2:
        best_rect = (bx + inset_px, by + inset_px, bw - inset_px * 2, bh - inset_px * 2)

    logger.info(
        "Canvas detected at x=%d y=%d w=%d h=%d (threshold=%d)",
        best_rect[0],
        best_rect[1],
        best_rect[2],
        best_rect[3],
        best_threshold,
    )

    return best_rect


def sample_bgr_patch(image: np.ndarray, center: tuple[int, int], radius: int = 2) -> tuple[int, int, int]:
    x, y = center
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(image.shape[1], x + radius + 1)
    y1 = min(image.shape[0], y + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError("Failed to sample palette color from screen")

    mean_bgr = np.rint(patch.mean(axis=(0, 1))).astype(np.int32)
    blue, green, red = mean_bgr.tolist()
    return blue, green, red


def compute_palette_colors(
    screen: np.ndarray,
    rect: tuple[int, int, int, int],
    template: np.ndarray,
) -> list[PaletteColor]:
    x, y, w, h = rect
    cell_count = max(1, round(template.shape[1] / max(template.shape[0], 1)))
    cell_w = w / float(cell_count)
    cy = int(y + h / 2.0)

    palette_colors: list[PaletteColor] = []
    for index in range(cell_count):
        cx = int(x + (index + 0.5) * cell_w)
        name = DEFAULT_PALETTE_NAMES[index] if index < len(DEFAULT_PALETTE_NAMES) else f"color_{index + 1}"
        bgr = sample_bgr_patch(screen, (cx, cy))
        palette_colors.append(PaletteColor(index=index, name=name, center=(cx, cy), bgr=bgr))

    return palette_colors


def bgr_to_luma(bgr: tuple[int, int, int]) -> float:
    blue, green, red = bgr
    return 0.114 * blue + 0.587 * green + 0.299 * red


def frame_to_grid(sampled_frame: np.ndarray, palette_colors: list[PaletteColor]) -> np.ndarray:
    palette_lookup = {color.index: color for color in palette_colors}
    draw_palette = [palette_lookup[index] for index in DRAW_PALETTE_INDICES if index in palette_lookup]
    if not draw_palette:
        raise RuntimeError("No drawable palette colors were detected")

    sampled_gray = cv2.cvtColor(sampled_frame, cv2.COLOR_BGR2GRAY).reshape((-1, 1)).astype(np.int32)
    palette_luma = np.array([[round(bgr_to_luma(color.bgr))] for color in draw_palette], dtype=np.int32)
    diffs = sampled_gray[:, None, :] - palette_luma[None, :, :]
    distances = np.sum(diffs * diffs, axis=2, dtype=np.int32)
    chosen = np.argmin(distances, axis=1)
    grid = np.array([draw_palette[idx].index for idx in chosen], dtype=np.int32)
    return grid.reshape((GRID_SIZE, GRID_SIZE))


def classify_luma_grid(luma_grid: np.ndarray, palette_colors: list[PaletteColor]) -> np.ndarray:
    palette_lookup = {color.index: color for color in palette_colors}
    draw_palette = [palette_lookup[index] for index in DRAW_PALETTE_INDICES if index in palette_lookup]
    if not draw_palette:
        raise RuntimeError("No drawable palette colors were detected")

    luma_values = luma_grid.reshape((-1, 1)).astype(np.int32)
    palette_luma = np.array([[round(bgr_to_luma(color.bgr))] for color in draw_palette], dtype=np.int32)
    diffs = luma_values[:, None, :] - palette_luma[None, :, :]
    distances = np.sum(diffs * diffs, axis=2, dtype=np.int32)
    chosen = np.argmin(distances, axis=1)
    grid = np.array([draw_palette[idx].index for idx in chosen], dtype=np.int32)
    return grid.reshape((GRID_SIZE, GRID_SIZE))


def render_grid_with_palette_colors(
    grid: np.ndarray,
    palette_colors: list[PaletteColor],
) -> np.ndarray:
    rendered = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    known = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for color in palette_colors:
        mask = grid == color.index
        if np.any(mask):
            rendered[mask] = np.array(color.bgr, dtype=np.uint8)
            known |= mask

    # Mark unexpected color indices clearly in debug output.
    if np.any(~known):
        rendered[~known] = np.array((255, 0, 255), dtype=np.uint8)

    return rendered


def observed_canvas_to_grid(
    screenshot_frame: np.ndarray,
    canvas_rect: tuple[int, int, int, int],
    palette_colors: list[PaletteColor],
) -> np.ndarray:
    x, y, w, h = canvas_rect
    canvas_crop = screenshot_frame[y : y + h, x : x + w]
    if canvas_crop.size == 0:
        raise RuntimeError("Canvas crop is empty while building observed grid")

    canvas_gray = cv2.cvtColor(canvas_crop, cv2.COLOR_BGR2GRAY)
    observed_luma = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    inner_margin_ratio = 0.2
    percentile = 85.0

    for row in range(GRID_SIZE):
        y0 = round(row * h / GRID_SIZE)
        y1 = round((row + 1) * h / GRID_SIZE)
        for col in range(GRID_SIZE):
            x0 = round(col * w / GRID_SIZE)
            x1 = round((col + 1) * w / GRID_SIZE)

            cell = canvas_gray[y0:y1, x0:x1]
            if cell.size == 0:
                raise RuntimeError("Observed canvas cell crop is empty")

            inset_x = min((cell.shape[1] - 1) // 2, round(cell.shape[1] * inner_margin_ratio))
            inset_y = min((cell.shape[0] - 1) // 2, round(cell.shape[0] * inner_margin_ratio))
            inner = cell[inset_y : cell.shape[0] - inset_y, inset_x : cell.shape[1] - inset_x]
            if inner.size == 0:
                inner = cell

            observed_luma[row, col] = round(float(np.percentile(inner, percentile)))

    return classify_luma_grid(observed_luma, palette_colors)


def validate_and_correct_frame(
    session: ScrcpySession,
    canvas_rect: tuple[int, int, int, int],
    expected_grid: np.ndarray,
    palette_colors: list[PaletteColor],
    action_delay_sec: float,
    frame_wait_timeout_sec: float,
    selected_color: int | None,
    frame_touch_actions: list[TouchAction],
    debug_dir: Path,
    frame_index: int,
    initial_shot: np.ndarray,
) -> tuple[int | None, np.ndarray, np.ndarray]:
    shot = initial_shot
    observed_grid = observed_canvas_to_grid(shot, canvas_rect, palette_colors)

    for attempt in range(1, VALIDATION_MAX_ATTEMPTS + 1):
        mismatch_mask = observed_grid != expected_grid
        mismatch_count = int(np.count_nonzero(mismatch_mask))
        if mismatch_count == 0:
            if attempt > 1:
                logger.info("Frame %d validated after %d correction pass(es)", frame_index, attempt - 1)
            return selected_color, shot, observed_grid

        logger.warning(
            "Frame %d validation mismatch after draw: attempt=%d mismatched_pixels=%d",
            frame_index,
            attempt,
            mismatch_count,
        )
        mismatch_path = save_mismatch_debug_image(
            debug_dir,
            frame_index,
            attempt,
            shot,
            canvas_rect,
            expected_grid,
            observed_grid,
            mismatch_mask,
            palette_colors,
        )
        logger.warning("Saved mismatch debug image: %s", mismatch_path)

        selected_color = draw_mask_scanline(
            session.client.control,
            canvas_rect,
            expected_grid,
            mismatch_mask,
            palette_colors,
            action_delay_sec,
            selected_color,
            frame_touch_actions,
        )

        time.sleep(action_delay_sec)
        shot = session.get_frame(timeout_sec=frame_wait_timeout_sec)
        observed_grid = observed_canvas_to_grid(shot, canvas_rect, palette_colors)

    mismatch_count = int(np.count_nonzero(observed_grid != expected_grid))
    failure_path = save_debug_screenshot(debug_dir, shot, f"validation_failed_frame_{frame_index:06d}")
    raise RuntimeError(
        "Frame validation failed after "
        f"{VALIDATION_MAX_ATTEMPTS} correction passes for frame {frame_index}; "
        f"remaining_mismatched_pixels={mismatch_count}; screenshot={failure_path}"
    )


def sample_frame_for_grid(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    target = GRID_SIZE
    h, w = frame.shape[:2]

    scale = max(target / w, target / h)
    resized_w = round(w * scale)
    resized_h = round(h * scale)

    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = (resized_w - target) // 2
    y0 = (resized_h - target) // 2
    cropped = resized[y0 : y0 + target, x0 : x0 + target]

    src_x0 = max(0, min(w - 1, round(x0 / scale)))
    src_y0 = max(0, min(h - 1, round(y0 / scale)))
    src_x1 = max(src_x0 + 1, min(w, round((x0 + target) / scale)))
    src_y1 = max(src_y0 + 1, min(h, round((y0 + target) / scale)))

    return cropped, (src_x0, src_y0, src_x1, src_y1)


def cell_center(canvas_rect: tuple[int, int, int, int], row: int, col: int) -> tuple[int, int]:
    x, y, w, h = canvas_rect
    cell_w = w / GRID_SIZE
    cell_h = h / GRID_SIZE
    px = int(x + (col + 0.5) * cell_w)
    py = int(y + (row + 0.5) * cell_h)
    return px, py


def tap(control: scrcpy.control.ControlSender, x: int, y: int) -> None:
    control.touch(x, y, scrcpy.ACTION_DOWN)
    control.touch(x, y, scrcpy.ACTION_UP)


def select_color(
    control: scrcpy.control.ControlSender,
    color: PaletteColor,
    selected_color: int | None,
    action_delay_sec: float,
) -> int:
    if selected_color == color.index:
        return selected_color

    tap(control, color.center[0], color.center[1])
    time.sleep(action_delay_sec)
    return color.index


def draw_mask_scanline(
    control: scrcpy.control.ControlSender,
    canvas_rect: tuple[int, int, int, int],
    target_grid: np.ndarray,
    draw_mask: np.ndarray,
    palette_colors: list[PaletteColor],
    action_delay_sec: float,
    selected_color: int | None,
    touch_actions: list[TouchAction],
) -> int | None:
    def _draw_scanline_runs(color_mask: np.ndarray) -> None:
        for row in range(GRID_SIZE):
            col = 0
            while col < GRID_SIZE:
                if not color_mask[row, col]:
                    col += 1
                    continue

                run_start = col
                run_end = col

                col += 1
                while col < GRID_SIZE and color_mask[row, col] and col == run_end + 1:
                    run_end = col
                    col += 1

                start_x, start_y = cell_center(canvas_rect, row, run_start)
                end_x, end_y = cell_center(canvas_rect, row, run_end)
                run_length = run_end - run_start + 1

                if run_length < 4:
                    for tap_col in range(run_start, run_end + 1):
                        tap_x, tap_y = cell_center(canvas_rect, row, tap_col)
                        tap(control, tap_x, tap_y)
                        touch_actions.append(TouchAction(kind="tap", start=(tap_x, tap_y), end=(tap_x, tap_y)))
                        time.sleep(action_delay_sec)
                else:
                    control.swipe(start_x, start_y, end_x, end_y)
                    touch_actions.append(
                        TouchAction(kind="swipe", start=(start_x, start_y), end=(end_x, end_y))
                    )

                time.sleep(action_delay_sec)

    active_colors = [color for color in palette_colors if color.index in DRAW_PALETTE_INDICES]
    color_masks: dict[int, np.ndarray] = {}
    present_colors: list[PaletteColor] = []
    for color in active_colors:
        color_mask = np.logical_and(draw_mask, target_grid == color.index)
        if np.any(color_mask):
            color_masks[color.index] = color_mask
            present_colors.append(color)

    if not present_colors:
        return selected_color

    phase_order = sorted(present_colors, key=lambda color: bgr_to_luma(color.bgr), reverse=True)

    for color in phase_order:
        selected_color = select_color(
            control,
            color,
            selected_color,
            action_delay_sec,
        )
        _draw_scanline_runs(color_masks[color.index])

    return selected_color


def clear_canvas(
    session: ScrcpySession,
    clear_button_center: tuple[int, int],
    clear_confirm_template: np.ndarray,
    score_threshold: float,
    action_delay_sec: float,
    confirm_timeout_sec: float,
    debug_dir: Path,
) -> None:
    control = session.client.control
    logger.info("Clearing canvas")

    tap(control, clear_button_center[0], clear_button_center[1])
    time.sleep(action_delay_sec)

    confirm_begin = time.time()
    deadline = confirm_begin + confirm_timeout_sec
    confirm_center: tuple[int, int] | None = None
    last_frame: np.ndarray | None = None

    while time.time() < deadline:
        frame = session.get_frame(timeout_sec=1.0)
        last_frame = frame
        try:
            x, y, w, h, _ = match_template_multiscale(frame, clear_confirm_template, score_threshold)
            confirm_center = (x + w // 2, y + h // 2)
            break
        except RuntimeError:
            time.sleep(0.05)

    if confirm_center is None:
        if last_frame is not None:
            save_debug_screenshot(debug_dir, last_frame, "detect_clear_confirm_failed")
        raise RuntimeError("Clear confirmation button was not detected")

    tap(control, confirm_center[0], confirm_center[1])
    confirm_transition_duration = time.time() - confirm_begin
    time.sleep(action_delay_sec + confirm_transition_duration)
    logger.info("Canvas cleared successfully")


def extract_sampled_frames_and_grids(
    video_path: Path, target_fps: float, palette_colors: list[PaletteColor]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    logger.info("Sampling source video at %.2f fps", target_fps)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = target_fps

    sampled: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    next_sample_time = 0.0
    target_step = 1.0 / target_fps
    frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_time = frame_idx / source_fps
        if frame_time + 1e-6 >= next_sample_time:
            sampled_frame, _ = sample_frame_for_grid(frame)
            grid = frame_to_grid(sampled_frame, palette_colors)
            sampled.append((frame.copy(), sampled_frame, grid))
            next_sample_time += target_step

        frame_idx += 1

    capture.release()

    if not sampled:
        raise RuntimeError("No frames were sampled from source video")

    logger.info("Sampled %d frames from source video", len(sampled))

    return sampled


def write_frame_image(path: Path, frame_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write screenshot: {path}")


def save_frame_comparison_debug_image(
    debug_dir: Path,
    frame_index: int,
    source_frame: np.ndarray,
    sampled_frame: np.ndarray,
    drawn_palette_frame: np.ndarray,
    screenshot_frame: np.ndarray,
    canvas_rect: tuple[int, int, int, int],
    touch_actions: list[TouchAction],
) -> None:
    x, y, w, h = canvas_rect

    sx0 = max(0, x)
    sy0 = max(0, y)
    sx1 = min(screenshot_frame.shape[1], x + w)
    sy1 = min(screenshot_frame.shape[0], y + h)
    if sx1 <= sx0 or sy1 <= sy0:
        raise RuntimeError("Canvas crop is out of screenshot bounds")

    canvas_crop = screenshot_frame[sy0:sy1, sx0:sx1]
    canvas_overlay = canvas_crop.copy()
    target_h = canvas_overlay.shape[0]

    # Overlay expected cell boundaries in the canvas pane.
    for col in range(1, GRID_SIZE):
        gx = round(col * canvas_overlay.shape[1] / GRID_SIZE)
        cv2.line(canvas_overlay, (gx, 0), (gx, canvas_overlay.shape[0] - 1), (160, 160, 160), 1, cv2.LINE_AA)

    for row in range(1, GRID_SIZE):
        gy = round(row * canvas_overlay.shape[0] / GRID_SIZE)
        cv2.line(canvas_overlay, (0, gy), (canvas_overlay.shape[1] - 1, gy), (160, 160, 160), 1, cv2.LINE_AA)

    # Overlay touch actions onto the canvas pane.
    for action in touch_actions:
        start_local = (action.start[0] - x, action.start[1] - y)
        end_local = (action.end[0] - x, action.end[1] - y)
        if action.kind == "tap":
            cv2.circle(canvas_overlay, start_local, 3, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.line(canvas_overlay, start_local, end_local, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas_overlay, start_local, 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas_overlay, end_local, 2, (0, 255, 255), -1, cv2.LINE_AA)

    source_with_rect = source_frame.copy()
    rx0, ry0, rx1, ry1 = sample_frame_for_grid(source_frame)[1]
    cv2.rectangle(source_with_rect, (rx0, ry0), (rx1 - 1, ry1 - 1), (0, 255, 0), 2)

    def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
        if image.shape[0] == height:
            return image
        new_w = max(1, round(image.shape[1] * (height / image.shape[0])))
        return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)

    left = _resize_to_height(source_with_rect, target_h)
    middle = _resize_to_height(sampled_frame, target_h)
    third = _resize_to_height(drawn_palette_frame, target_h)
    right = _resize_to_height(canvas_overlay, target_h)

    separator = np.full((target_h, 3, 3), (255, 0, 0), dtype=np.uint8)
    combined = np.hstack((left, separator, middle, separator, third, separator, right))

    frame_debug_dir = debug_dir / "frames"
    frame_debug_dir.mkdir(parents=True, exist_ok=True)
    path = frame_debug_dir / f"frame_{frame_index:06d}.png"
    ok = cv2.imwrite(str(path), combined)
    if not ok:
        raise RuntimeError(f"Failed to write frame debug image: {path}")


def save_mismatch_debug_image(
    debug_dir: Path,
    frame_index: int,
    attempt: int,
    screenshot_frame: np.ndarray,
    canvas_rect: tuple[int, int, int, int],
    expected_grid: np.ndarray,
    observed_grid: np.ndarray,
    mismatch_mask: np.ndarray,
    palette_colors: list[PaletteColor],
) -> Path:
    x, y, w, h = canvas_rect
    sx0 = max(0, x)
    sy0 = max(0, y)
    sx1 = min(screenshot_frame.shape[1], x + w)
    sy1 = min(screenshot_frame.shape[0], y + h)
    if sx1 <= sx0 or sy1 <= sy0:
        raise RuntimeError("Canvas crop is out of screenshot bounds")

    canvas_crop = screenshot_frame[sy0:sy1, sx0:sx1]
    canvas_overlay = canvas_crop.copy()
    expected_frame = render_grid_with_palette_colors(expected_grid, palette_colors)
    observed_frame = render_grid_with_palette_colors(observed_grid, palette_colors)
    target_h = canvas_overlay.shape[0]

    for row in range(GRID_SIZE):
        y0 = round(row * canvas_overlay.shape[0] / GRID_SIZE)
        y1 = round((row + 1) * canvas_overlay.shape[0] / GRID_SIZE)
        for col in range(GRID_SIZE):
            x0 = round(col * canvas_overlay.shape[1] / GRID_SIZE)
            x1 = round((col + 1) * canvas_overlay.shape[1] / GRID_SIZE)
            if mismatch_mask[row, col]:
                cv2.rectangle(canvas_overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
            else:
                cv2.rectangle(canvas_overlay, (x0, y0), (x1 - 1, y1 - 1), (96, 96, 96), 1)

    def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
        if image.shape[0] == height:
            return image
        new_w = max(1, round(image.shape[1] * (height / image.shape[0])))
        return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_NEAREST)

    expected_panel = _resize_to_height(expected_frame, target_h)
    observed_panel = _resize_to_height(observed_frame, target_h)
    canvas_panel = _resize_to_height(canvas_overlay, target_h)

    separator = np.full((target_h, 3, 3), (255, 0, 0), dtype=np.uint8)
    combined = np.hstack((expected_panel, separator, observed_panel, separator, canvas_panel))

    frame_debug_dir = debug_dir / "frames"
    frame_debug_dir.mkdir(parents=True, exist_ok=True)
    path = frame_debug_dir / f"frame_{frame_index:06d}_attempt_{attempt:02d}_mismatch.png"
    ok = cv2.imwrite(str(path), combined)
    if not ok:
        raise RuntimeError(f"Failed to write mismatch debug image: {path}")
    return path


def source_has_audio(ffprobe_path: str, source_video: Path) -> bool:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(source_video),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def compile_output_video(config: Config, source_video: Path) -> None:
    logger.info("Compiling output video: %s", config.output_video_path)
    image_input = str(config.screenshots_dir / "frame_%06d.png")

    if source_has_audio(config.ffprobe_path, source_video):
        cmd = [
            config.ffmpeg_path,
            "-y",
            "-framerate",
            str(config.draw_fps),
            "-i",
            image_input,
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(config.output_video_path),
        ]
    else:
        cmd = [
            config.ffmpeg_path,
            "-y",
            "-framerate",
            str(config.draw_fps),
            "-i",
            image_input,
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(config.output_video_path),
        ]

    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    logger.info("Output video generated: %s", config.output_video_path)


def load_templates(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    palette_path = config.templates_dir / "palette-first-row.png"
    clear_path = config.templates_dir / "clear-button.png"
    clear_confirm_path = config.templates_dir / "clear-confirm-button.png"

    palette = cv2.imread(str(palette_path), cv2.IMREAD_COLOR)
    clear_button = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
    clear_confirm = cv2.imread(str(clear_confirm_path), cv2.IMREAD_COLOR)

    if palette is None or clear_button is None or clear_confirm is None:
        raise RuntimeError("One or more template images could not be loaded")

    logger.info("Template images loaded")

    return palette, clear_button, clear_confirm


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_config_from_json(project_root)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting automation")

    ensure_adb_connection(config)

    source_video = get_first_video_file(config.assets_dir)
    palette_template, clear_template, clear_confirm_template = load_templates(config)

    clear_screenshots_dir(config.screenshots_dir)
    clear_debug_frames_dir(config.debug_dir)

    session = ScrcpySession(serial=config.emulator_serial, max_fps=int(max(30, config.draw_fps * 2)))
    logger.info("Starting scrcpy session")
    session.start()

    try:
        screen = session.get_frame(timeout_sec=config.frame_wait_timeout_sec)
        logger.info("Received initial video frame")

        try:
            canvas_rect = detect_canvas_rect(screen)
        except RuntimeError:
            save_debug_screenshot(config.debug_dir, screen, "detect_canvas_failed")
            raise

        try:
            px, py, pw, ph, score = match_template_multiscale(
                screen,
                palette_template,
                config.template_score_threshold,
            )
        except RuntimeError:
            save_debug_screenshot(config.debug_dir, screen, "detect_palette_failed")
            raise
        palette_rect = (px, py, pw, ph)
        logger.info("Palette detected with score %.3f", score)

        try:
            cx, cy, cw, ch, score = match_template_multiscale(
                screen,
                clear_template,
                config.template_score_threshold,
            )
        except RuntimeError:
            save_debug_screenshot(config.debug_dir, screen, "detect_clear_button_failed")
            raise
        clear_button_rect = (cx, cy, cw, ch)
        clear_button_center = (cx + cw // 2, cy + ch // 2)
        logger.info("Clear button detected with score %.3f", score)

        palette_colors = compute_palette_colors(screen, palette_rect, palette_template)
        logger.info(
            "Palette colors resolved: %s",
            ", ".join(
                f"{color.name}#{color.index}@{color.center}={color.bgr}" for color in palette_colors
            ),
        )
        draw_color_indices = [color.index for color in palette_colors if color.index in DRAW_PALETTE_INDICES]
        if BACKGROUND_PALETTE_INDEX not in draw_color_indices:
            raise RuntimeError("Background palette color was not detected")

        save_base_debug_image(
            config.debug_dir,
            screen,
            canvas_rect,
            palette_rect,
            clear_button_rect,
        )

        sampled_frames = extract_sampled_frames_and_grids(
            source_video,
            config.draw_fps,
            palette_colors,
        )

        total_frame_count = len(sampled_frames)
        start_idx = 0 if config.frame_start is None else max(0, config.frame_start)
        if config.frame_end is not None:
            end_idx = min(total_frame_count, config.frame_end + 1)
            if end_idx < start_idx:
                raise ValueError("frame_end must be greater than or equal to frame_start")
        else:
            end_idx = total_frame_count

        sampled_frames = sampled_frames[start_idx:end_idx]
        if start_idx != 0 or end_idx != total_frame_count:
            logger.info(
                "Frame range configured: processing frames %d to %d (%d frames)",
                start_idx,
                end_idx - 1,
                len(sampled_frames),
            )

        clear_canvas(
            session,
            clear_button_center,
            clear_confirm_template,
            config.template_score_threshold,
            config.action_delay_sec,
            config.clear_confirm_timeout_sec,
            config.debug_dir,
        )

        selected_color: int | None = None
        prev_grid: np.ndarray | None = None

        total_frames = len(sampled_frames)
        logger.info("Starting draw loop for %d frames", total_frames)
        for i, (source_frame, sampled_frame, grid) in enumerate(sampled_frames, start=1):
            if prev_grid is None:
                draw_mask = grid != BACKGROUND_PALETTE_INDEX
                logger.info("Frame %d/%d: initial full draw", i, total_frames)
            else:
                changed_mask = grid != prev_grid
                changed_count = int(np.count_nonzero(changed_mask))
                current_draw_count = int(np.count_nonzero(np.isin(grid, draw_color_indices)))

                if changed_count < current_draw_count:
                    draw_mask = changed_mask
                    logger.debug(
                        "Frame %d/%d: incremental draw changed=%d drawn=%d",
                        i,
                        total_frames,
                        changed_count,
                        current_draw_count,
                    )
                else:
                    logger.info(
                        "Frame %d/%d: clear and redraw changed=%d drawn=%d",
                        i,
                        total_frames,
                        changed_count,
                        current_draw_count,
                    )
                    clear_canvas(
                        session,
                        clear_button_center,
                        clear_confirm_template,
                        config.template_score_threshold,
                        config.action_delay_sec,
                        config.clear_confirm_timeout_sec,
                        config.debug_dir,
                    )
                    selected_color = None
                    draw_mask = np.isin(grid, draw_color_indices)

            frame_touch_actions: list[TouchAction] = []
            selected_color = draw_mask_scanline(
                session.client.control,
                canvas_rect,
                grid,
                draw_mask,
                palette_colors,
                config.action_delay_sec,
                selected_color,
                frame_touch_actions,
            )

            time.sleep(config.action_delay_sec)
            shot = session.get_frame(timeout_sec=config.frame_wait_timeout_sec)
            selected_color, shot, prev_grid = validate_and_correct_frame(
                session,
                canvas_rect,
                grid,
                palette_colors,
                config.action_delay_sec,
                config.frame_wait_timeout_sec,
                selected_color,
                frame_touch_actions,
                config.debug_dir,
                i,
                shot,
            )
            screenshot_path = config.screenshots_dir / f"frame_{i:06d}.png"
            write_frame_image(screenshot_path, shot)
            drawn_palette_frame = render_grid_with_palette_colors(grid, palette_colors)
            save_frame_comparison_debug_image(
                config.debug_dir,
                i,
                source_frame,
                sampled_frame,
                drawn_palette_frame,
                shot,
                canvas_rect,
                frame_touch_actions,
            )

            if i == 1 or i == total_frames or i % 10 == 0:
                logger.info("Saved screenshot for frame %d/%d", i, total_frames)

        compile_output_video(config, source_video)
        logger.info("Automation completed successfully")

    except Exception:
        logger.exception("Automation failed")
        raise
    finally:
        logger.info("Stopping scrcpy session")
        session.stop()


if __name__ == "__main__":
    main()
