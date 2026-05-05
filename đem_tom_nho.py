from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


os.environ["YOLO_AUTOINSTALL"] = "False"

MODEL_PATH = Path(__file__).with_name("nano.pt")
IMAGE_FILE_TYPES = [
    ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
    ("All files", "*.*"),
]
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

DEFAULT_COLS = 6
DEFAULT_ROWS = 4
DEFAULT_IMGSZ = 960
DEFAULT_MIN_IMGSZ = 640
DEFAULT_CONF = 0.12
DEFAULT_OVERLAP_RATIO = 0.35
DEFAULT_NMS_IOU = 0.30
DEFAULT_CENTER_DISTANCE = 6.0
DEFAULT_GPU_BATCH_SIZE = 4
DEFAULT_CPU_BATCH_SIZE = 1


def read_image_unicode(path: str | Path) -> np.ndarray | None:
    """Read image paths safely, including Unicode paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def choose_device() -> int | str:
    """Use CUDA when available, otherwise CPU."""
    if torch is not None and torch.cuda.is_available():
        return 0
    return "cpu"


def describe_device(device: int | str) -> str:
    """Return a user-friendly device label for the UI."""
    if torch is not None and device != "cpu" and torch.cuda.is_available():
        try:
            return f"CUDA - {torch.cuda.get_device_name(int(device))}"
        except Exception:
            return "CUDA GPU"
    if torch is None:
        return "CPU - torch unavailable"
    return "CPU - torch khong co CUDA"


def configure_runtime(device: int | str) -> None:
    """Enable safe runtime optimizations when CUDA is available."""
    if torch is None or device == "cpu":
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def is_cuda_device(device: int | str) -> bool:
    """Check whether the current inference device is CUDA."""
    return bool(torch is not None and device != "cpu" and torch.cuda.is_available())


def make_divisible(value: int, divisor: int = 32) -> int:
    """Round up to the nearest model-friendly stride."""
    return max(divisor, ((int(value) + divisor - 1) // divisor) * divisor)


def choose_inference_imgsz(tile_w: int, tile_h: int, overlap_x: int, overlap_y: int) -> int:
    """Avoid sending small tiles to an unnecessarily large imgsz."""
    tile_input_side = max(tile_w + overlap_x * 2, tile_h + overlap_y * 2)
    target = min(DEFAULT_IMGSZ, max(DEFAULT_MIN_IMGSZ, tile_input_side))
    return make_divisible(target)


def warmup_inference(model: YOLO, device: int | str) -> str:
    """Verify the selected runtime by running one tiny inference."""
    warmup_imgsz = DEFAULT_MIN_IMGSZ
    warmup_image = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
    use_half = is_cuda_device(device)
    inference_context = torch.inference_mode() if torch is not None else nullcontext()

    with inference_context:
        model.predict(
            source=warmup_image,
            imgsz=warmup_imgsz,
            conf=DEFAULT_CONF,
            verbose=False,
            device=device,
            half=use_half,
        )

    if use_half and torch is not None:
        torch.cuda.synchronize()
        try:
            memory_mb = torch.cuda.memory_allocated(int(device)) / (1024 * 1024)
            return f"{describe_device(device)} | warmup CUDA OK | VRAM ~{memory_mb:.0f} MB"
        except Exception:
            pass

    return describe_device(device)


def predict_tiles_in_batches(
    model: YOLO,
    tiles: list[np.ndarray],
    imgsz: int,
    conf: float,
    device: int | str,
    use_half: bool,
    batch_size: int,
):
    """Run YOLO on a list of tiles with small GPU-friendly batches."""
    results = []
    inference_context = torch.inference_mode() if torch is not None else nullcontext()

    with inference_context:
        for start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[start:start + batch_size]
            source = batch_tiles[0] if len(batch_tiles) == 1 else batch_tiles

            try:
                batch_results = model.predict(
                    source=source,
                    imgsz=imgsz,
                    conf=conf,
                    verbose=False,
                    device=device,
                    half=use_half,
                    batch=len(batch_tiles),
                )
            except RuntimeError as exc:
                if not is_cuda_device(device) or "out of memory" not in str(exc).lower() or len(batch_tiles) == 1:
                    raise

                torch.cuda.empty_cache()
                batch_results = []
                for tile in batch_tiles:
                    single_result = model.predict(
                        source=tile,
                        imgsz=imgsz,
                        conf=conf,
                        verbose=False,
                        device=device,
                        half=use_half,
                    )
                    batch_results.append(single_result[0])

            if len(batch_tiles) == 1 and batch_results:
                results.append(batch_results[0])
            else:
                results.extend(batch_results)

    return results


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(
    detections: list[dict[str, object]],
    iou_threshold: float,
) -> list[dict[str, object]]:
    """Remove overlapping duplicate detections."""
    remaining = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict[str, object]] = []

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            item
            for item in remaining
            if compute_iou(best["bbox"], item["bbox"]) < iou_threshold
        ]

    return kept


def deduplicate_by_center_distance(
    detections: list[dict[str, object]],
    distance_threshold: float,
) -> list[dict[str, object]]:
    """Merge leftover duplicates without collapsing nearby distinct shrimp."""
    if distance_threshold <= 0:
        return detections

    sorted_detections = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict[str, object]] = []

    for item in sorted_detections:
        x1, y1, x2, y2 = item["bbox"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        is_duplicate = False
        for kept_item in kept:
            kx1, ky1, kx2, ky2 = kept_item["bbox"]
            kept_center_x = (kx1 + kx2) / 2.0
            kept_center_y = (ky1 + ky2) / 2.0

            dx = center_x - kept_center_x
            dy = center_y - kept_center_y
            if dx * dx + dy * dy >= distance_threshold * distance_threshold:
                continue

            # Only merge when the two boxes still look like the same shrimp.
            iou = compute_iou(item["bbox"], kept_item["bbox"])
            center_in_kept = kx1 <= center_x <= kx2 and ky1 <= center_y <= ky2
            kept_center_in_item = x1 <= kept_center_x <= x2 and y1 <= kept_center_y <= y2
            if iou >= 0.05 or center_in_kept or kept_center_in_item:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(item)

    return kept


def center_belongs_to_core(
    center_x: float,
    center_y: float,
    core_x1: int,
    core_y1: int,
    core_x2: int,
    core_y2: int,
    col: int,
    row: int,
    cols: int,
    rows: int,
) -> bool:
    """Assign each object to exactly one core tile."""
    if col == cols - 1:
        inside_x = core_x1 <= center_x <= core_x2
    else:
        inside_x = core_x1 <= center_x < core_x2

    if row == rows - 1:
        inside_y = core_y1 <= center_y <= core_y2
    else:
        inside_y = core_y1 <= center_y < core_y2

    return inside_x and inside_y


def draw_tile_grid(
    image: np.ndarray,
    cols: int,
    rows: int,
    tile_w: int,
    tile_h: int,
    width: int,
    height: int,
) -> None:
    """Draw the 24-tile grid and tile labels."""
    grid_color = (255, 180, 0)
    label_bg = (0, 0, 0)
    label_fg = (255, 255, 255)

    for col in range(1, cols):
        x = col * tile_w
        cv2.line(image, (x, 0), (x, height), grid_color, 2)

    for row in range(1, rows):
        y = row * tile_h
        cv2.line(image, (0, y), (width, y), grid_color, 2)

    for row in range(rows):
        for col in range(cols):
            x1 = col * tile_w
            y1 = row * tile_h
            x2 = (col + 1) * tile_w if col < cols - 1 else width
            y2 = (row + 1) * tile_h if row < rows - 1 else height
            label = f"O {row + 1}-{col + 1}"

            label_right = min(x1 + 92, x2 - 6)
            label_bottom = min(y1 + 30, y2 - 6)
            if label_right <= x1 + 6 or label_bottom <= y1 + 6:
                continue

            cv2.rectangle(image, (x1 + 6, y1 + 6), (label_right, label_bottom), label_bg, -1)
            cv2.putText(
                image,
                label,
                (x1 + 10, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                label_fg,
                1,
                cv2.LINE_AA,
            )


def run_tiled_count(
    model: YOLO,
    image: np.ndarray,
    progress_callback=None,
    device: int | str = "cpu",
) -> dict[str, object]:
    """Run tiled YOLO inference and deduplicate detections on the full image."""
    cols = DEFAULT_COLS
    rows = DEFAULT_ROWS
    conf = DEFAULT_CONF
    overlap_ratio = DEFAULT_OVERLAP_RATIO
    nms_iou_threshold = DEFAULT_NMS_IOU
    center_distance_threshold = DEFAULT_CENTER_DISTANCE

    h, w = image.shape[:2]
    tile_w = max(1, w // cols)
    tile_h = max(1, h // rows)

    overlap_x = min(tile_w // 2, max(16, int(tile_w * overlap_ratio)))
    overlap_y = min(tile_h // 2, max(16, int(tile_h * overlap_ratio)))
    imgsz = choose_inference_imgsz(tile_w, tile_h, overlap_x, overlap_y)

    all_detections: list[dict[str, object]] = []
    total_tiles = cols * rows
    current_tile = 0
    use_half = is_cuda_device(device)
    batch_size = DEFAULT_GPU_BATCH_SIZE if use_half else DEFAULT_CPU_BATCH_SIZE
    tile_jobs: list[dict[str, object]] = []

    for row in range(rows):
        for col in range(cols):
            core_x1 = col * tile_w
            core_y1 = row * tile_h
            core_x2 = (col + 1) * tile_w if col < cols - 1 else w
            core_y2 = (row + 1) * tile_h if row < rows - 1 else h

            tile_x1 = max(0, core_x1 - overlap_x)
            tile_y1 = max(0, core_y1 - overlap_y)
            tile_x2 = min(w, core_x2 + overlap_x)
            tile_y2 = min(h, core_y2 + overlap_y)

            tile_jobs.append(
                {
                    "col": col,
                    "row": row,
                    "core_x1": core_x1,
                    "core_y1": core_y1,
                    "core_x2": core_x2,
                    "core_y2": core_y2,
                    "tile_x1": tile_x1,
                    "tile_y1": tile_y1,
                    "tile": image[tile_y1:tile_y2, tile_x1:tile_x2],
                }
            )

    batch_results = predict_tiles_in_batches(
        model=model,
        tiles=[job["tile"] for job in tile_jobs],
        imgsz=imgsz,
        conf=conf,
        device=device,
        use_half=use_half,
        batch_size=batch_size,
    )

    for job, result in zip(tile_jobs, batch_results):
        col = int(job["col"])
        row = int(job["row"])
        core_x1 = int(job["core_x1"])
        core_y1 = int(job["core_y1"])
        core_x2 = int(job["core_x2"])
        core_y2 = int(job["core_y2"])
        tile_x1 = int(job["tile_x1"])
        tile_y1 = int(job["tile_y1"])
        boxes = result.boxes
        raw_count = len(boxes)
        kept_count = 0

        if raw_count:
            xyxy = boxes.xyxy.int().cpu().numpy()
            scores = boxes.conf.cpu().numpy()

            xyxy[:, [0, 2]] += tile_x1
            xyxy[:, [1, 3]] += tile_y1
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)

            valid_mask = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
            if np.any(valid_mask):
                centers_x = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
                centers_y = (xyxy[:, 1] + xyxy[:, 3]) * 0.5

                if col == cols - 1:
                    inside_x = (centers_x >= core_x1) & (centers_x <= core_x2)
                else:
                    inside_x = (centers_x >= core_x1) & (centers_x < core_x2)

                if row == rows - 1:
                    inside_y = (centers_y >= core_y1) & (centers_y <= core_y2)
                else:
                    inside_y = (centers_y >= core_y1) & (centers_y < core_y2)

                keep_mask = valid_mask & inside_x & inside_y
                kept_boxes = xyxy[keep_mask]
                kept_scores = scores[keep_mask]
                kept_count = int(len(kept_boxes))

                for coords, score in zip(kept_boxes.tolist(), kept_scores.tolist()):
                    gx1, gy1, gx2, gy2 = coords
                    all_detections.append(
                        {
                            "bbox": (gx1, gy1, gx2, gy2),
                            "score": float(score),
                        }
                    )

        current_tile += 1
        if progress_callback is not None:
            progress_callback(current_tile, total_tiles, raw_count, kept_count)

    filtered = non_max_suppression(all_detections, nms_iou_threshold)
    filtered = deduplicate_by_center_distance(filtered, center_distance_threshold)

    return {
        "count": len(filtered),
        "detections": filtered,
        "cols": cols,
        "rows": rows,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "width": w,
        "height": h,
        "imgsz": imgsz,
    }


def render_result_image(
    image: np.ndarray,
    detections: list[dict[str, object]],
    count: int,
    cols: int,
    rows: int,
    tile_w: int,
    tile_h: int,
    width: int,
    height: int,
    show_boxes: bool,
) -> np.ndarray:
    """Build the result image with optional bounding boxes."""
    canvas = image.copy()

    if show_boxes:
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (18, 18), (500, 86), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    cv2.putText(
        canvas,
        f"TOTAL SHRIMP: {count}",
        (34, 62),
        cv2.FONT_HERSHEY_DUPLEX,
        1.15,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    return canvas


class ZoomableImageCanvas:
    """Canvas with cursor-centered zoom, drag pan, fit, and 100% view."""

    def __init__(self, parent: tk.Widget, empty_text: str, zoom_callback=None):
        self.frame = tk.Frame(parent, bg="#f8fbff")
        self.canvas = tk.Canvas(
            self.frame,
            bg="#eef4fb",
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
        )
        self.h_scroll = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set,
        )

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.zoom_callback = zoom_callback
        self.cv_image: np.ndarray | None = None
        self.photo_image = None
        self.image_id = None
        self.zoom_level = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 12.0
        self.display_width = 1
        self.display_height = 1
        self.empty_text = empty_text
        self.placeholder_id = self.canvas.create_text(
            320,
            240,
            text=empty_text,
            fill="#64748b",
            font=("Segoe UI", 15, "bold"),
        )

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def notify_zoom(self) -> None:
        if callable(self.zoom_callback):
            self.zoom_callback(self.zoom_level)

    def set_image(self, image: np.ndarray, preserve_view: bool = False) -> None:
        old_xview = self.canvas.xview()
        old_yview = self.canvas.yview()
        old_zoom = self.zoom_level

        self.cv_image = image.copy()
        if self.placeholder_id is not None:
            self.canvas.delete(self.placeholder_id)
            self.placeholder_id = None

        if preserve_view and self.image_id is not None:
            self.zoom_level = old_zoom
            self.display_image()
            self.restore_view(old_xview, old_yview)
            self.notify_zoom()
        else:
            self.frame.after(10, self.fit_to_screen)

    def clear(self) -> None:
        self.cv_image = None
        self.photo_image = None
        self.zoom_level = 1.0
        self.display_width = 1
        self.display_height = 1

        if self.image_id is not None:
            self.canvas.delete(self.image_id)
            self.image_id = None

        if self.placeholder_id is None:
            self.placeholder_id = self.canvas.create_text(
                320,
                240,
                text=self.empty_text,
                fill="#64748b",
                font=("Segoe UI", 15, "bold"),
            )

        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.notify_zoom()

    def on_canvas_resize(self, event) -> None:
        if self.placeholder_id is not None:
            self.canvas.coords(self.placeholder_id, event.width / 2, event.height / 2)

    def display_image(self) -> None:
        if self.cv_image is None:
            return

        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        new_w = max(1, int(pil_image.width * self.zoom_level))
        new_h = max(1, int(pil_image.height * self.zoom_level))
        pil_image = pil_image.resize((new_w, new_h), RESAMPLE)

        self.display_width = new_w
        self.display_height = new_h
        self.photo_image = ImageTk.PhotoImage(pil_image)

        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        else:
            self.canvas.itemconfig(self.image_id, image=self.photo_image)

        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

    def restore_view(self, xview: tuple[float, float], yview: tuple[float, float]) -> None:
        self.canvas.xview_moveto(xview[0] if xview else 0)
        self.canvas.yview_moveto(yview[0] if yview else 0)

    def fit_to_screen(self) -> None:
        if self.cv_image is None:
            return

        canvas_width = max(200, self.canvas.winfo_width())
        canvas_height = max(200, self.canvas.winfo_height())
        image_h, image_w = self.cv_image.shape[:2]

        scale_x = (canvas_width * 0.95) / image_w
        scale_y = (canvas_height * 0.95) / image_h
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, min(scale_x, scale_y)))
        self.display_image()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.notify_zoom()

    def set_actual_size(self) -> None:
        self.set_zoom(1.0)

    def set_zoom(self, new_zoom: float, anchor_x: float | None = None, anchor_y: float | None = None) -> None:
        if self.cv_image is None:
            return

        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        if abs(new_zoom - self.zoom_level) < 1e-9:
            return

        if anchor_x is None:
            anchor_x = self.canvas.winfo_width() / 2
        if anchor_y is None:
            anchor_y = self.canvas.winfo_height() / 2

        old_zoom = self.zoom_level
        image_x = self.canvas.canvasx(anchor_x) / max(old_zoom, 1e-9)
        image_y = self.canvas.canvasy(anchor_y) / max(old_zoom, 1e-9)

        self.zoom_level = new_zoom
        self.display_image()
        self.scroll_point_to_anchor(image_x, image_y, anchor_x, anchor_y)
        self.notify_zoom()

    def scroll_point_to_anchor(
        self,
        image_x: float,
        image_y: float,
        anchor_x: float,
        anchor_y: float,
    ) -> None:
        total_w = max(1, self.display_width)
        total_h = max(1, self.display_height)
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())

        target_canvas_x = image_x * self.zoom_level
        target_canvas_y = image_y * self.zoom_level

        if total_w <= view_w:
            self.canvas.xview_moveto(0)
        else:
            left = min(max(0, target_canvas_x - anchor_x), total_w - view_w)
            self.canvas.xview_moveto(left / total_w)

        if total_h <= view_h:
            self.canvas.yview_moveto(0)
        else:
            top = min(max(0, target_canvas_y - anchor_y), total_h - view_h)
            self.canvas.yview_moveto(top / total_h)

    def on_mouse_wheel(self, event) -> None:
        if self.cv_image is None:
            return

        if getattr(event, "num", None) == 4 or event.delta > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15

        self.set_zoom(self.zoom_level * factor, event.x, event.y)

    def on_pan_start(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_double_click(self, _event) -> None:
        self.fit_to_screen()


class ShrimpCounterApp:
    """GUI for side-by-side shrimp counting results."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dem tom AI")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(1360, max(1180, screen_w - 60))
        window_h = min(780, max(700, screen_h - 120))
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(1100, 680)

        self.colors = {
            "app_bg": "#eaf0f6",
            "card_bg": "#fcfdff",
            "border": "#d9e4f0",
            "panel_bg": "#f4f8fc",
            "text_primary": "#16324f",
            "text_muted": "#66788f",
            "blue": "#2563eb",
            "blue_dark": "#1d4ed8",
            "blue_soft": "#e8f1ff",
            "green": "#15803d",
            "green_dark": "#166534",
            "green_soft": "#e8f7ed",
            "orange": "#c66d15",
            "orange_dark": "#a85e12",
            "orange_soft": "#fff3e5",
            "slate_soft": "#f8fafc",
            "disabled_bg": "#c8d6e5",
            "disabled_fg": "#eef4fb",
        }

        self.root.configure(bg=self.colors["app_bg"])
        self.configure_ttk_styles()

        if not MODEL_PATH.exists():
            messagebox.showerror("Loi", f"Khong tim thay model:\n{MODEL_PATH}")
            self.root.destroy()
            return

        self.device = choose_device()
        self.device_label = describe_device(self.device)
        configure_runtime(self.device)
        self.model = YOLO(str(MODEL_PATH))
        try:
            self.model.fuse()
        except Exception:
            pass
        try:
            self.runtime_label = warmup_inference(self.model, self.device)
        except Exception as exc:
            messagebox.showerror(
                "Loi khoi tao thiet bi",
                f"Khong the khoi tao runtime tren {self.device_label}:\n{exc}",
            )
            self.root.destroy()
            return

        self.image_path: str | None = None
        self.original_image: np.ndarray | None = None
        self.detections: list[dict[str, object]] = []
        self.result_image: np.ndarray | None = None
        self.last_result_meta: dict[str, object] | None = None
        self.render_cache: dict[bool, np.ndarray] = {}

        self.count_var = tk.StringVar(value="--")
        self.time_var = tk.StringVar(value="--")
        self.progress_text_var = tk.StringVar(value=f"0 / {DEFAULT_COLS * DEFAULT_ROWS} o")
        self.status_var = tk.StringVar(value="San sang.")
        self.show_boxes_var = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0.0)

        self.build_ui()

    def configure_ttk_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Shrimp.Horizontal.TProgressbar",
            thickness=12,
            troughcolor="#dbe7f5",
            background=self.colors["blue"],
            bordercolor="#dbe7f5",
            lightcolor=self.colors["blue"],
            darkcolor=self.colors["blue"],
        )

    def create_badge(self, parent: tk.Widget, text: str, bg: str, fg: str) -> tk.Label:
        badge = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        )
        return badge

    def create_action_button(
        self,
        parent: tk.Widget,
        text: str,
        role: str,
        command,
        width: int = 15,
    ) -> tk.Button:
        button = tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            padx=16,
            pady=10,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
            activeforeground="white",
            disabledforeground=self.colors["disabled_fg"],
            highlightthickness=0,
        )
        button._role = role
        self.set_button_enabled(button, True)
        return button

    def create_tool_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=6,
            padx=10,
            pady=7,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            bg="#eef4ff",
            fg=self.colors["blue_dark"],
            activebackground="#dbeafe",
            activeforeground=self.colors["blue_dark"],
            highlightthickness=0,
        )

    def set_button_enabled(self, button: tk.Button, enabled: bool) -> None:
        role = getattr(button, "_role", "primary")
        palette = {
            "primary": (self.colors["blue"], self.colors["blue_dark"]),
            "success": (self.colors["green"], self.colors["green_dark"]),
        }
        bg, active_bg = palette.get(role, palette["primary"])
        if enabled:
            button.config(
                state=tk.NORMAL,
                bg=bg,
                fg="white",
                activebackground=active_bg,
                cursor="hand2",
            )
        else:
            button.config(
                state=tk.DISABLED,
                bg=self.colors["disabled_bg"],
                fg=self.colors["disabled_fg"],
                activebackground=self.colors["disabled_bg"],
                cursor="arrow",
            )

    def set_boxes_toggle_enabled(self, enabled: bool) -> None:
        fg = self.colors["text_primary"] if enabled else "#93a5ba"
        self.show_boxes_check.config(
            state=tk.NORMAL if enabled else tk.DISABLED,
            fg=fg,
            activeforeground=fg,
            cursor="hand2" if enabled else "arrow",
        )

    def create_metric_card(
        self,
        parent: tk.Widget,
        title: str,
        value_var: tk.StringVar,
        bg: str,
        accent: str,
        detail_text: str | None = None,
        detail_var: tk.StringVar | None = None,
    ) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )

        body = tk.Frame(card, bg=bg)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        tk.Label(
            body,
            text=title,
            font=("Segoe UI", 9, "bold"),
            bg=bg,
            fg=accent,
            anchor=tk.W,
        ).pack(fill=tk.X)

        tk.Label(
            body,
            textvariable=value_var,
            font=("Segoe UI Semibold", 13),
            bg=bg,
            fg=self.colors["text_primary"],
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=320,
        ).pack(fill=tk.X, pady=(6, 2))

        if detail_var is not None or detail_text:
            if detail_var is not None:
                detail_kwargs = {"textvariable": detail_var}
            else:
                detail_kwargs = {"text": detail_text}

            tk.Label(
                body,
                font=("Segoe UI", 9),
                bg=bg,
                fg=self.colors["text_muted"],
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=320,
                **detail_kwargs,
            ).pack(fill=tk.X)

        return card

    def build_ui(self) -> None:
        container = tk.Frame(self.root, bg=self.colors["app_bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        header = tk.Frame(
            container,
            bg=self.colors["card_bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        header.pack(fill=tk.X)

        header_body = tk.Frame(header, bg=self.colors["card_bg"])
        header_body.pack(fill=tk.X, padx=14, pady=14)

        title_row = tk.Frame(header_body, bg=self.colors["card_bg"])
        title_row.pack(fill=tk.X)

        title_block = tk.Frame(title_row, bg=self.colors["card_bg"])
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            title_block,
            text="Dem tom AI",
            font=("Segoe UI Semibold", 16),
            bg=self.colors["card_bg"],
            fg=self.colors["text_primary"],
        ).pack(anchor=tk.W)

        action_row = tk.Frame(header_body, bg=self.colors["card_bg"])
        action_row.pack(fill=tk.X, pady=(12, 10))

        left_actions = tk.Frame(action_row, bg=self.colors["card_bg"])
        left_actions.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.choose_button = self.create_action_button(
            left_actions,
            "Chon anh",
            "primary",
            self.choose_image,
        )
        self.choose_button.pack(side=tk.LEFT, padx=(0, 10))

        self.count_button = self.create_action_button(
            left_actions,
            "Dem tom",
            "success",
            self.run_count,
        )
        self.count_button.pack(side=tk.LEFT, padx=(0, 12))
        self.set_button_enabled(self.count_button, False)

        toggle_box = tk.Frame(
            left_actions,
            bg=self.colors["slate_soft"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        toggle_box.pack(side=tk.LEFT, padx=(0, 10))

        self.show_boxes_check = tk.Checkbutton(
            toggle_box,
            text="Hien bounding box",
            variable=self.show_boxes_var,
            bg=self.colors["slate_soft"],
            fg=self.colors["text_primary"],
            activebackground=self.colors["slate_soft"],
            activeforeground=self.colors["text_primary"],
            selectcolor=self.colors["card_bg"],
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self.toggle_boxes,
            state=tk.DISABLED,
        )
        self.show_boxes_check.pack()
        self.set_boxes_toggle_enabled(False)

        metrics = tk.Frame(header_body, bg=self.colors["card_bg"])
        metrics.pack(fill=tk.X)
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_columnconfigure(1, weight=1)
        metrics.grid_columnconfigure(2, weight=1)

        count_card = self.create_metric_card(
            metrics,
            "So tom",
            self.count_var,
            self.colors["green_soft"],
            self.colors["green_dark"],
        )
        count_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        time_card = self.create_metric_card(
            metrics,
            "Thoi gian",
            self.time_var,
            self.colors["orange_soft"],
            self.colors["orange_dark"],
        )
        time_card.grid(row=0, column=1, sticky="nsew", padx=(0, 8))

        progress_card = self.create_metric_card(
            metrics,
            "Tien trinh",
            self.progress_text_var,
            self.colors["blue_soft"],
            self.colors["blue_dark"],
        )
        progress_card.grid(row=0, column=2, sticky="nsew")

        status_row = tk.Frame(header_body, bg=self.colors["card_bg"])
        status_row.pack(fill=tk.X, pady=(10, 0))

        status_box = tk.Frame(
            status_row,
            bg=self.colors["panel_bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        status_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))

        self.status_label = tk.Label(
            status_box,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=self.colors["panel_bg"],
            fg=self.colors["text_primary"],
            anchor=tk.W,
            padx=12,
            pady=10,
        )
        self.status_label.pack(fill=tk.X)

        progress_box = tk.Frame(
            status_row,
            bg=self.colors["card_bg"],
        )
        progress_box.pack(side=tk.RIGHT)

        tk.Label(
            progress_box,
            text="Tien trinh quet",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text_muted"],
            anchor=tk.E,
        ).pack(anchor="e", pady=(0, 4))

        self.progress_bar = ttk.Progressbar(
            progress_box,
            variable=self.progress_var,
            maximum=DEFAULT_COLS * DEFAULT_ROWS,
            length=220,
            mode="determinate",
            style="Shrimp.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(anchor="e")

        viewer = tk.PanedWindow(
            container,
            orient=tk.HORIZONTAL,
            bg=self.colors["app_bg"],
            bd=0,
            sashwidth=8,
            sashrelief=tk.FLAT,
            opaqueresize=True,
        )
        viewer.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left_panel = tk.Frame(
            viewer,
            bg=self.colors["card_bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        right_panel = tk.Frame(
            viewer,
            bg=self.colors["card_bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        viewer.add(left_panel, minsize=340, stretch="always")
        viewer.add(right_panel, minsize=340, stretch="always")

        self.original_canvas = self.build_panel(
            left_panel,
            "Anh goc",
            "Chon anh de xem ban goc",
        )
        self.result_canvas = self.build_panel(
            right_panel,
            "Anh ket qua",
            "Ket qua dem se hien o day",
        )

    def build_panel(
        self,
        panel: tk.Frame,
        title: str,
        empty_text: str,
    ) -> ZoomableImageCanvas:
        top = tk.Frame(panel, bg=self.colors["card_bg"])
        top.pack(fill=tk.X, padx=12, pady=(10, 8))

        tk.Label(
            top,
            text=title,
            font=("Segoe UI Semibold", 12),
            bg=self.colors["card_bg"],
            fg=self.colors["text_primary"],
        ).pack(side=tk.LEFT)

        canvas_holder = tk.Frame(panel, bg=self.colors["card_bg"])
        canvas_holder.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        canvas_shell = tk.Frame(
            canvas_holder,
            bg=self.colors["panel_bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        canvas_shell.pack(fill=tk.BOTH, expand=True)

        canvas = ZoomableImageCanvas(canvas_shell, empty_text)
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(title="Chon anh tom", filetypes=IMAGE_FILE_TYPES)
        if not path:
            return

        image = read_image_unicode(path)
        if image is None:
            messagebox.showerror("Loi", "Khong doc duoc anh da chon.")
            return

        self.image_path = path
        self.original_image = image
        self.detections = []
        self.result_image = None
        self.last_result_meta = None
        self.render_cache.clear()

        self.original_canvas.set_image(image)
        self.result_canvas.clear()
        self.count_var.set("--")
        self.time_var.set("--")
        self.progress_text_var.set(f"0 / {DEFAULT_COLS * DEFAULT_ROWS} o")
        self.progress_var.set(0)
        self.status_var.set("Da chon anh. Bam 'Dem tom' de bat dau.")
        self.count_button.config(text="Dem tom")
        self.set_button_enabled(self.count_button, True)
        self.set_boxes_toggle_enabled(False)

    def on_progress(self, current_tile: int, total_tiles: int, raw_count: int, kept_count: int) -> None:
        self.progress_var.set(current_tile)
        self.progress_text_var.set(f"{current_tile} / {total_tiles} o")
        self.status_var.set(
            f"Dang xu ly o {current_tile}/{total_tiles} | raw={raw_count} | kept={kept_count}"
        )
        self.root.update_idletasks()

    def refresh_result_image(self, preserve_view: bool) -> None:
        if self.original_image is None or self.last_result_meta is None:
            return

        show_boxes = bool(self.show_boxes_var.get())
        cached_image = self.render_cache.get(show_boxes)
        if cached_image is None:
            cached_image = render_result_image(
                image=self.original_image,
                detections=self.detections,
                count=int(self.last_result_meta["count"]),
                cols=int(self.last_result_meta["cols"]),
                rows=int(self.last_result_meta["rows"]),
                tile_w=int(self.last_result_meta["tile_w"]),
                tile_h=int(self.last_result_meta["tile_h"]),
                width=int(self.last_result_meta["width"]),
                height=int(self.last_result_meta["height"]),
                show_boxes=show_boxes,
            )
            self.render_cache[show_boxes] = cached_image

        self.result_image = cached_image
        self.result_canvas.set_image(self.result_image, preserve_view=preserve_view)

    def toggle_boxes(self) -> None:
        if self.last_result_meta is None:
            return
        self.refresh_result_image(preserve_view=True)

    def run_count(self) -> None:
        if self.original_image is None:
            messagebox.showwarning("Canh bao", "Hay chon anh truoc.")
            return

        self.set_button_enabled(self.choose_button, False)
        self.count_button.config(text="Dang dem...")
        self.set_button_enabled(self.count_button, False)
        self.set_boxes_toggle_enabled(False)
        self.progress_var.set(0)
        self.progress_text_var.set(f"0 / {DEFAULT_COLS * DEFAULT_ROWS} o")
        self.status_var.set("Dang dem tom...")
        self.root.update_idletasks()

        if is_cuda_device(self.device):
            torch.cuda.synchronize()
        start_time = time.time()

        try:
            result = run_tiled_count(
                model=self.model,
                image=self.original_image,
                progress_callback=self.on_progress,
                device=self.device,
            )

            if is_cuda_device(self.device):
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            self.detections = list(result["detections"])
            self.last_result_meta = result
            self.render_cache.clear()

            self.refresh_result_image(preserve_view=False)

            total_shrimp = int(result["count"])
            self.count_var.set(str(total_shrimp))
            self.time_var.set(f"{elapsed:.2f}s")
            self.progress_text_var.set(f"{DEFAULT_COLS * DEFAULT_ROWS} / {DEFAULT_COLS * DEFAULT_ROWS} o")
            self.progress_var.set(DEFAULT_COLS * DEFAULT_ROWS)
            self.status_var.set(f"Hoan thanh. Da dem {total_shrimp} con trong {elapsed:.2f}s.")
            self.set_boxes_toggle_enabled(True)
        except Exception as exc:
            messagebox.showerror("Loi xu ly", str(exc))
            self.status_var.set("Xu ly that bai.")
        finally:
            self.set_button_enabled(self.choose_button, True)
            self.count_button.config(text="Dem tom")
            self.set_button_enabled(self.count_button, self.original_image is not None)


def main() -> None:
    root = tk.Tk()
    app = ShrimpCounterApp(root)
    if root.winfo_exists():
        root.mainloop()


if __name__ == "__main__":
    main()
