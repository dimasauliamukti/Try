import sys
import os
import json
import numpy as np
import cv2
import onnxruntime as ort
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QFrame, QMessageBox,
    QSizePolicy
)
from config_keypoints import MODEL_BBOX_DIR
from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal, QRectF
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont, QCursor,
    QWheelEvent
)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from segmentation_mp import MediaPipeSkinSegmenter

SELFIE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'selfie_multiclass.tflite')

DEFAULT_BBOX_MODEL = MODEL_BBOX_DIR / "good_augwed/weights/best.onnx"


class Segmenter:
    def __init__(self):
        self.mp_selfie = MediaPipeSkinSegmenter(str(SELFIE_MODEL_PATH))

    def segment(self, image_bgr: np.ndarray):
        mask, image_seg, skin_mask, _, _ = self.mp_selfie.segment(image_bgr)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        return mask, image_seg, skin_mask


def extract_contour_points(mask: np.ndarray, max_points: int = 600):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    if len(pts) > max_points:
        idx = np.round(np.linspace(0, len(pts) - 1, max_points)).astype(int)
        pts = pts[idx]
    return [(int(p[0]), int(p[1])) for p in pts]


class BboxDetector:
    """Wrapper ONNX untuk deteksi bounding box neck."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name  = self.session.get_inputs()[0].name
        inp_shape        = self.session.get_inputs()[0].shape
        self.input_h = inp_shape[2] if isinstance(inp_shape[2], int) else 640
        self.input_w = inp_shape[3] if isinstance(inp_shape[3], int) else 640

    def detect(self, image: np.ndarray, padding: int = 10):
        orig_h, orig_w = image.shape[:2]

        resized = cv2.resize(image, (self.input_w, self.input_h))
        inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]

        output_raw = self.session.run(None, {self.input_name: inp})[0]
        output = output_raw[0].T

        if len(output) == 0:
            return None, None, 0.0

        best_idx = int(output[:, 4].argmax())
        best     = output[best_idx]
        cx, cy, w, h = best[0], best[1], best[2], best[3]
        score = float(best[4])

        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        x1 = max(0,      int((cx - w / 2) * scale_x) - padding)
        y1 = max(0,      int((cy - h / 2) * scale_y) - padding)
        x2 = min(orig_w, int((cx + w / 2) * scale_x) + padding)
        y2 = min(orig_h, int((cy + h / 2) * scale_y) + padding)

        if x2 <= x1 or y2 <= y1:
            return None, None, score

        crop = image[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2), score


def crop_to_bbox(image: np.ndarray, detector: BboxDetector, padding: int = 10):
    crop, coords, score = detector.detect(image, padding=padding)
    if crop is None:
        print(f"[INFO] No bounding box detected (score={score:.3f}).")
        return None, None
    print(f"[INFO] Bbox detected with score={score:.3f}, coords={coords}")
    return crop, coords


def filter_and_offset_contour(contour_pts, crop_box):
    x1, y1, x2, y2 = crop_box
    return [
        (px - x1, py - y1)
        for (px, py) in contour_pts
        if x1 <= px <= x2 and y1 <= py <= y2
    ]


def export_yolo_keypoints(keypoints, img_w, img_h, image_path, out_path, crop_box=None):
    parts = ["0 0.500000 0.500000 1.000000 1.000000"]
    for kp in keypoints:
        if kp is None:
            parts += ["0.000000", "0.000000", "0"]
        else:
            parts += [f"{kp[0]/img_w:.6f}", f"{kp[1]/img_h:.6f}", "2"]

    with open(out_path, "w") as f:
        f.write(" ".join(parts))

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        meta = {
            "image_original": os.path.basename(image_path),
            "crop_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "crop_size": {"w": int(img_w), "h": int(img_h)},
            "keypoints": []
        }
        for i, kp in enumerate(keypoints):
            if kp is None:
                meta["keypoints"].append({
                    "id": i + 1,
                    "label": KEYPOINT_LABELS[i],
                    "crop_relative": None,
                    "original_absolute": None
                })
            else:
                meta["keypoints"].append({
                    "id": i + 1,
                    "label": KEYPOINT_LABELS[i],
                    "crop_relative": [round(kp[0] / img_w, 6), round(kp[1] / img_h, 6)],
                    "original_absolute": [int(x1 + kp[0]), int(y1 + kp[1])]
                })

        meta_path = os.path.splitext(out_path)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[META] Saved annotation metadata -> {meta_path}")


KEYPOINT_COLORS = [QColor(255, 80, 80), QColor(80, 200, 80)]
KEYPOINT_LABELS = ["Right (neck-shoulder)", "Left  (neck-shoulder)"]
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.1, 8.0, 1.15


class AnnotationCanvas(QWidget):
    status_message    = pyqtSignal(str)
    keypoints_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.WheelFocus)

        self._base_image   = None
        self._contour_pts  = []
        self._keypoints    = [None, None]
        self._next_kp_idx  = 0

        self._overlay_visible   = True
        self._keypoints_visible = True
        self._snap_to_contour   = True

        self._zoom       = 1.0
        self._pan        = QPointF(0, 0)
        self._pan_start  = None
        self._pan_origin = None
        self._base_pix   = None

        self._cropped_image  = None
        self._original_image = None
        self._crop_box       = None
        self._showing_original = False

    def load(self, image_bgr, contour_pts):
        self._base_image  = image_bgr.copy()
        self._contour_pts = contour_pts
        self._keypoints   = [None, None]
        self._next_kp_idx = 0
        self._showing_original = False
        self._build_base_pix()
        self._fit_to_view()
        self.update()
        self.status_message.emit(
            f"{len(contour_pts)} contour points. Click for KP-1 (right).")
        self.keypoints_changed.emit()

    def set_context(self, cropped_image, original_image, crop_box):
        self._cropped_image  = cropped_image.copy()
        self._original_image = original_image.copy()
        self._crop_box       = crop_box
        self._showing_original = False

    def clear_context(self):
        self._cropped_image  = None
        self._original_image = None
        self._crop_box       = None
        self._showing_original = False

    def can_toggle_context(self):
        return self._cropped_image is not None and self._original_image is not None

    def is_showing_original(self):
        return self._showing_original

    def toggle_context_view(self):
        if not self.can_toggle_context():
            return False
        self._showing_original = not self._showing_original

        if self._showing_original:
            display_img = self._build_original_display()
            self._base_image = display_img
            self._contour_pts_saved = self._contour_pts
            self._contour_pts = []
        else:
            self._base_image = self._cropped_image.copy()
            if hasattr(self, '_contour_pts_saved'):
                self._contour_pts = self._contour_pts_saved

        self._build_base_pix()
        self._fit_to_view()
        self.update()
        return True

    def _build_original_display(self):
        img = self._original_image.copy()
        x1, y1, x2, y2 = self._crop_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 2)

        if self._keypoints_visible:
            kp_colors_bgr = [(80, 80, 255), (80, 200, 80)]
            for i, kp in enumerate(self._keypoints):
                if kp is not None:
                    ox, oy = kp[0] + x1, kp[1] + y1
                    cv2.circle(img, (ox, oy), 6, kp_colors_bgr[i], -1)
                    cv2.circle(img, (ox, oy), 7, (255, 255, 255), 1)
                    cv2.putText(img, f"KP{i+1}", (ox + 8, oy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if hasattr(self, '_contour_pts_saved') and self._contour_pts_saved and self._overlay_visible:
            for (px, py) in self._contour_pts_saved:
                cv2.circle(img, (px + x1, py + y1), 1, (255, 100, 0), -1)

        return img

    def set_image_only(self, image_bgr):
        self._base_image  = image_bgr.copy()
        self._contour_pts = []
        self._keypoints   = [None, None]
        self._next_kp_idx = 0
        self._showing_original = False
        self.clear_context()
        self._build_base_pix()
        self._fit_to_view()
        self.update()
        self.keypoints_changed.emit()

    def toggle_overlay(self):
        self._overlay_visible = not self._overlay_visible
        if self._showing_original:
            self._base_image = self._build_original_display()
        self._build_base_pix()
        self.update()

    def toggle_keypoints(self):
        self._keypoints_visible = not self._keypoints_visible
        if self._showing_original:
            self._base_image = self._build_original_display()
            self._build_base_pix()
        self.update()

    def set_snap_to_contour(self, enabled: bool):
        self._snap_to_contour = enabled

    def reset_kp(self, idx):
        self._keypoints[idx] = None
        self._next_kp_idx = next(
            (i for i, k in enumerate(self._keypoints) if k is None), 2)
        if self._showing_original:
            self._base_image = self._build_original_display()
            self._build_base_pix()
        self.update()
        self.status_message.emit(
            f"KP{idx+1} reset. Click to place KP{idx+1} again.")
        self.keypoints_changed.emit()

    def zoom_in(self):
        self._apply_zoom(ZOOM_STEP, self._center())

    def zoom_out(self):
        self._apply_zoom(1 / ZOOM_STEP, self._center())

    def zoom_reset(self):
        self._fit_to_view()
        self.update()

    def get_keypoints(self):
        return list(self._keypoints)

    def get_image_size(self):
        if self._cropped_image is not None:
            h, w = self._cropped_image.shape[:2]
            return w, h
        if self._base_image is None:
            return None
        h, w = self._base_image.shape[:2]
        return w, h

    def load_keypoints(self, keypoints):
        for i in range(min(2, len(keypoints))):
            self._keypoints[i] = keypoints[i]
        self._next_kp_idx = next(
            (i for i, k in enumerate(self._keypoints) if k is None), 2)
        if self._showing_original:
            self._base_image = self._build_original_display()
            self._build_base_pix()
        self.update()
        self.keypoints_changed.emit()

    def _center(self):
        return QPointF(self.width() / 2, self.height() / 2)

    def _fit_to_view(self):
        if self._base_pix is None:
            return
        pw, ph = self._base_pix.width(), self._base_pix.height()
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        self._zoom = min(ww / pw, wh / ph)
        self._pan  = QPointF((ww - pw * self._zoom) / 2,
                              (wh - ph * self._zoom) / 2)

    def _apply_zoom(self, factor, pivot):
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))
        rf = new_zoom / self._zoom
        self._pan = QPointF(
            pivot.x() - rf * (pivot.x() - self._pan.x()),
            pivot.y() - rf * (pivot.y() - self._pan.y()))
        self._zoom = new_zoom
        self.update()

    def _widget_to_image(self, wx, wy):
        return ((wx - self._pan.x()) / self._zoom,
                (wy - self._pan.y()) / self._zoom)

    def _build_base_pix(self):
        if self._base_image is None:
            self._base_pix = None
            return
        img = self._base_image.copy()
        if not self._showing_original and self._overlay_visible and self._contour_pts:
            for (px, py) in self._contour_pts:
                cv2.circle(img, (px, py), 1, (255, 100, 0), -1)
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self._base_pix = QPixmap.fromImage(qimg)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 24))

        if self._base_pix is None:
            painter.setPen(QColor(100, 100, 120))
            painter.setFont(QFont("Consolas", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No image loaded.\nClick 'Select Images'.")
            return

        pw = self._base_pix.width()  * self._zoom
        ph = self._base_pix.height() * self._zoom
        target = QRectF(self._pan.x(), self._pan.y(), pw, ph)
        painter.drawPixmap(target, self._base_pix, QRectF(self._base_pix.rect()))

        if self._keypoints_visible and not self._showing_original:
            painter.setRenderHint(QPainter.Antialiasing)
            for i, kp in enumerate(self._keypoints):
                if kp is None:
                    continue
                sx    = kp[0] * self._zoom + self._pan.x()
                sy    = kp[1] * self._zoom + self._pan.y()
                sp    = QPointF(sx, sy)
                color = KEYPOINT_COLORS[i]
                r     = max(1, int(3 * min(self._zoom, 2)))
                painter.setPen(QPen(Qt.white, 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(sp, r, r)
                painter.setPen(QPen(Qt.white))
                painter.setFont(QFont("Consolas",
                                      max(3, int(4 * min(self._zoom, 2))),
                                      QFont.Bold))
                painter.drawText(QPointF(sx + r + 3, sy + 5), f"KP{i+1}")

        painter.setPen(QColor(160, 160, 180))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(8, self.height() - 8, f"zoom {self._zoom:.2f}x")

        snap_color = QColor(80, 200, 120) if self._snap_to_contour else QColor(255, 140, 0)
        snap_text  = "SNAP: contour" if self._snap_to_contour else "SNAP: free"
        painter.setPen(snap_color)
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.drawText(8, self.height() - 22, snap_text)

        if self._showing_original:
            painter.setPen(QColor(255, 165, 0))
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.drawText(8, 18, "CONTEXT MODE — click disabled")

    def wheelEvent(self, event):
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self._apply_zoom(factor, QPointF(event.pos()))

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_start  = event.pos()
            self._pan_origin = QPointF(self._pan)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.LeftButton:
            self._handle_click(event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan = self._pan_origin + QPointF(delta)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_start  = None
            self._pan_origin = None
            self.setCursor(QCursor(Qt.CrossCursor))

    def resizeEvent(self, event):
        if self._base_pix is not None:
            self._fit_to_view()
        super().resizeEvent(event)

    def _handle_click(self, wx, wy):
        if self._showing_original:
            return
        if self._base_image is None:
            return
        if self._next_kp_idx >= 2:
            self.status_message.emit(
                "Both keypoints placed. Use Reset KP1 / Reset KP2 to change.")
            return

        ix, iy = self._widget_to_image(wx, wy)

        if self._snap_to_contour:
            if not self._contour_pts:
                self.status_message.emit(
                    "Snap mode ON but no contour available. "
                    "Run segmentation first or toggle snap OFF.")
                return
            pts_arr = np.array(self._contour_pts, dtype=float)
            dists   = np.hypot(pts_arr[:, 0] - ix, pts_arr[:, 1] - iy)
            placed  = self._contour_pts[int(np.argmin(dists))]
        else:
            h, w = self._base_image.shape[:2]
            placed = (int(max(0, min(w - 1, round(ix)))),
                      int(max(0, min(h - 1, round(iy)))))

        self._keypoints[self._next_kp_idx] = placed
        label = KEYPOINT_LABELS[self._next_kp_idx]
        self._next_kp_idx += 1
        self.update()
        self.keypoints_changed.emit()

        if self._next_kp_idx < 2:
            self.status_message.emit(
                f"KP '{label}' -> {placed}. "
                f"Click for '{KEYPOINT_LABELS[self._next_kp_idx]}'.")
        else:
            self.status_message.emit(
                f"KP '{label}' -> {placed}. "
                "Both keypoints done! Export when ready.")


DARK     = "#12121a"
PANEL    = "#1e1e2e"
ACCENT   = "#4f8ef7"
MUTED    = "#64748b"
BTN_TEXT = "#e2e8f0"


def _btn(text, color=ACCENT, min_w=180, h=34):
    b = QPushButton(text)
    b.setMinimumWidth(min_w)
    b.setFixedHeight(h)
    b.setStyleSheet(f"""
        QPushButton {{
            background: {color};
            color: {BTN_TEXT};
            border: none;
            border-radius: 6px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 11px;
            font-weight: 600;
            padding: 0 8px;
        }}
        QPushButton:hover   {{ background: {color}cc; }}
        QPushButton:pressed {{ background: {color}88; }}
        QPushButton:disabled {{ background: #2d2d3d; color: {MUTED}; }}
    """)
    return b


def _divider():
    d = QFrame()
    d.setFrameShape(QFrame.HLine)
    d.setStyleSheet("color: #2d2d3d;")
    return d


def _section(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {MUTED}; font-size: 9px; font-weight: 700; "
        "letter-spacing: 2px; margin-top: 4px;")
    return lbl


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ground Truth Keypoint Annotator")
        self.resize(1200, 820)
        self.setStyleSheet(f"background: {DARK}; color: #e2e8f0;")

        self._segmenter   = Segmenter()
        self._bbox_detector: BboxDetector | None = None

        self._image_paths = []
        self._current_idx = -1
        self._image_bgr   = None

        self._save_crop_dir      = None
        self._full_images_dir    = None   # ← subfolder full_images di dalam crop dir
        self._save_yolo_dir      = None
        self._annot_dir          = None
        self._current_crop_image = None
        self._current_crop_box   = None

        self._build_ui()
        self._load_bbox_model(str(DEFAULT_BBOX_MODEL))

    # ── BBOX MODEL ──────────────────────────────────────────────────────────

    def _load_bbox_model(self, path: str):
        if not path or not os.path.isfile(path):
            self.lbl_bbox_model.setText("Model: not found — use Change Bbox Model")
            return
        try:
            self._bbox_detector = BboxDetector(path)
            short = os.path.basename(path)
            self.lbl_bbox_model.setText(f"Active: {short}")
            self.status.showMessage(f"ONNX bbox model loaded: {short}")
        except Exception as e:
            self._bbox_detector = None
            self.lbl_bbox_model.setText("Model: load failed — use Change Bbox Model")
            self.status.showMessage(f"ONNX bbox model load error: {e}")

    # ── BUILD UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(
            f"background: {PANEL}; border-right: 1px solid #2d2d3d;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(12, 18, 12, 18)
        sb.setSpacing(7)

        title = QLabel("KP\nAnnotator")
        title.setStyleSheet(
            f"color: {ACCENT}; font-family: Consolas; "
            "font-size: 20px; font-weight: 700; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignCenter)
        sb.addWidget(title)
        sb.addWidget(_divider())

        # ── BBOX MODEL ──────────────────────────────────────────────────────
        sb.addWidget(_section("BBOX MODEL"))

        self.lbl_bbox_model = QLabel("Model: initializing...")
        self.lbl_bbox_model.setWordWrap(True)
        self.lbl_bbox_model.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        sb.addWidget(self.lbl_bbox_model)

        self.btn_load_bbox_model = _btn("📦  Change Bbox Model (.onnx)", "#6d28d9")
        self.btn_load_bbox_model.clicked.connect(self._on_load_bbox_model)
        sb.addWidget(self.btn_load_bbox_model)

        sb.addWidget(_divider())

        # ── IMAGES ──────────────────────────────────────────────────────────
        sb.addWidget(_section("IMAGES"))
        self.btn_load = _btn("📂  Select Images", ACCENT)
        self.btn_load.clicked.connect(self._on_load)
        sb.addWidget(self.btn_load)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self.btn_prev = _btn("◀ Prev", "#334155", 88)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next = _btn("Next ▶", "#334155", 88)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        sb.addLayout(nav_row)

        self.lbl_counter = QLabel("–")
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        self.lbl_counter.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; font-family: Consolas;")
        sb.addWidget(self.lbl_counter)

        self.lbl_filename = QLabel("–")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setFixedWidth(196)
        self.lbl_filename.setStyleSheet(
            "color: #e2e8f0; font-size: 12px; font-family: Consolas;"
            "background: #0d0d16; border: 1px solid #2d2d3d;"
            "border-radius: 8px; padding: 6px 8px;")
        sb.addWidget(self.lbl_filename)

        sb.addWidget(_divider())

        # ── SEGMENTATION ────────────────────────────────────────────────────
        sb.addWidget(_section("SEGMENTATION"))
        self.btn_segment = _btn("🔍  Segment + Crop Neck", "#0ea5e9")
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self._on_segment)
        sb.addWidget(self.btn_segment)

        sb.addWidget(_divider())

        # ── VIEW ────────────────────────────────────────────────────────────
        sb.addWidget(_section("VIEW"))
        self.btn_toggle_overlay = _btn("👁  Overlay: ON", "#7c3aed")
        self.btn_toggle_overlay.setEnabled(False)
        self.btn_toggle_overlay.clicked.connect(self._on_toggle_overlay)
        sb.addWidget(self.btn_toggle_overlay)

        self.btn_toggle_kp = _btn("📍  Keypoints: ON", "#7c3aed")
        self.btn_toggle_kp.setEnabled(False)
        self.btn_toggle_kp.clicked.connect(self._on_toggle_kp)
        sb.addWidget(self.btn_toggle_kp)

        self.btn_toggle_context = _btn("🔭  Context View: OFF", "#9333ea")
        self.btn_toggle_context.setEnabled(False)
        self.btn_toggle_context.clicked.connect(self._on_toggle_context)
        sb.addWidget(self.btn_toggle_context)

        sb.addWidget(_divider())

        # ── PLACEMENT MODE ──────────────────────────────────────────────────
        sb.addWidget(_section("PLACEMENT MODE"))
        self.btn_toggle_snap = _btn("🧲  Snap to Contour: ON", "#0f766e")
        self.btn_toggle_snap.clicked.connect(self._on_toggle_snap)
        sb.addWidget(self.btn_toggle_snap)

        snap_hint = QLabel("OFF = place anywhere on image")
        snap_hint.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        snap_hint.setAlignment(Qt.AlignCenter)
        sb.addWidget(snap_hint)

        sb.addWidget(_divider())

        # ── ZOOM ────────────────────────────────────────────────────────────
        sb.addWidget(_section("ZOOM"))
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(5)
        self.btn_zoom_in  = _btn("🔍+", "#1e3a5f", 55, 30)
        self.btn_zoom_out = _btn("🔍−", "#1e3a5f", 55, 30)
        self.btn_zoom_fit = _btn("Fit",  "#1e3a5f", 55, 30)
        self.btn_zoom_in.clicked.connect(lambda: self.canvas.zoom_in())
        self.btn_zoom_out.clicked.connect(lambda: self.canvas.zoom_out())
        self.btn_zoom_fit.clicked.connect(lambda: self.canvas.zoom_reset())
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_fit)
        sb.addLayout(zoom_row)

        hint = QLabel("Scroll=zoom  |  RMB drag=pan")
        hint.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        hint.setAlignment(Qt.AlignCenter)
        sb.addWidget(hint)

        sb.addWidget(_divider())

        # ── RESET KEYPOINTS ─────────────────────────────────────────────────
        sb.addWidget(_section("RESET KEYPOINTS"))
        self.btn_reset_kp1 = _btn("↺  Reset KP1 (Right)", "#b45309")
        self.btn_reset_kp1.setEnabled(False)
        self.btn_reset_kp1.clicked.connect(lambda: self._on_reset_kp(0))
        sb.addWidget(self.btn_reset_kp1)

        self.btn_reset_kp2 = _btn("↺  Reset KP2 (Left)", "#b45309")
        self.btn_reset_kp2.setEnabled(False)
        self.btn_reset_kp2.clicked.connect(lambda: self._on_reset_kp(1))
        sb.addWidget(self.btn_reset_kp2)

        sb.addWidget(_divider())

        # ── SAVE CROP ───────────────────────────────────────────────────────
        sb.addWidget(_section("SAVE CROP"))
        self.btn_set_crop_dir = _btn("📁  Set Crop Folder", "#0f766e")
        self.btn_set_crop_dir.clicked.connect(self._on_set_crop_dir)
        sb.addWidget(self.btn_set_crop_dir)

        self.lbl_crop_dir = QLabel("Folder: not set")
        self.lbl_crop_dir.setWordWrap(True)
        self.lbl_crop_dir.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        sb.addWidget(self.lbl_crop_dir)

        # Label info subfolder full_images
        self.lbl_full_images_dir = QLabel("")
        self.lbl_full_images_dir.setWordWrap(True)
        self.lbl_full_images_dir.setStyleSheet(
            f"color: #22c55e; font-size: 8px; font-family: Consolas;")
        sb.addWidget(self.lbl_full_images_dir)

        self.btn_save_crop = _btn("💾  Save Crop Image", "#0f766e")
        self.btn_save_crop.setEnabled(False)
        self.btn_save_crop.clicked.connect(self._on_save_crop)
        sb.addWidget(self.btn_save_crop)

        sb.addWidget(_divider())

        # ── EXPORT YOLO ─────────────────────────────────────────────────────
        sb.addWidget(_section("EXPORT YOLO"))
        self.btn_set_yolo_dir = _btn("📁  Set YOLO Folder", "#16a34a")
        self.btn_set_yolo_dir.clicked.connect(self._on_set_yolo_dir)
        sb.addWidget(self.btn_set_yolo_dir)

        self.lbl_yolo_dir = QLabel("Folder: not set")
        self.lbl_yolo_dir.setWordWrap(True)
        self.lbl_yolo_dir.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        sb.addWidget(self.lbl_yolo_dir)

        self.btn_export_yolo = _btn("💾  Export YOLO", "#16a34a")
        self.btn_export_yolo.setEnabled(False)
        self.btn_export_yolo.clicked.connect(self._on_export_yolo)
        sb.addWidget(self.btn_export_yolo)

        sb.addStretch()

        sb.addWidget(_divider())

        # ── LOAD ANNOTATION ─────────────────────────────────────────────────
        sb.addWidget(_section("LOAD ANNOTATION"))
        self.btn_set_annot_dir = _btn("📁  Set Annotation Folder", "#dc2626")
        self.btn_set_annot_dir.clicked.connect(self._on_set_annot_dir)
        sb.addWidget(self.btn_set_annot_dir)

        self.lbl_annot_dir = QLabel("Folder: not set")
        self.lbl_annot_dir.setWordWrap(True)
        self.lbl_annot_dir.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        sb.addWidget(self.lbl_annot_dir)

        self.btn_load_yolo = _btn("📥  Load Annotation + Map JSON", "#b91c1c")
        self.btn_load_yolo.setEnabled(False)
        self.btn_load_yolo.clicked.connect(self._on_load_yolo_annotation)
        sb.addWidget(self.btn_load_yolo)

        root.addWidget(sidebar)

        # ── CANVAS + KP OVERLAY ─────────────────────────────────────────────
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background: transparent;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = AnnotationCanvas()
        self.canvas.status_message.connect(self._set_status)
        self.canvas.keypoints_changed.connect(self._on_kp_changed)
        canvas_layout.addWidget(self.canvas)

        self.kp_overlay = QFrame(canvas_container)
        self.kp_overlay.setStyleSheet(
            "QFrame { background: rgba(13,13,22,160); border-radius: 8px;"
            "border: 1px solid rgba(100,100,140,120); }")
        self.kp_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        kp_ov_lay = QVBoxLayout(self.kp_overlay)
        kp_ov_lay.setContentsMargins(10, 6, 10, 6)
        kp_ov_lay.setSpacing(3)
        kp_hdr = QLabel("KEYPOINTS")
        kp_hdr.setStyleSheet(
            f"color: rgba(100,116,139,200); font-size: 8px; font-weight: 700;"
            "letter-spacing: 2px; background: transparent; border: none;")
        kp_ov_lay.addWidget(kp_hdr)
        self.kp_labels = []
        for i, color in enumerate(KEYPOINT_COLORS):
            lbl = QLabel(f"KP{i+1}: –")
            lbl.setStyleSheet(
                f"color: rgba({color.red()},{color.green()},{color.blue()},220);"
                "font-size: 11px; font-family: Consolas;"
                "background: transparent; border: none;")
            kp_ov_lay.addWidget(lbl)
            self.kp_labels.append(lbl)
        self.kp_overlay.adjustSize()
        self.kp_overlay.raise_()

        canvas_container.resizeEvent = self._reposition_kp_overlay
        root.addWidget(canvas_container, stretch=1)

        self.status = QStatusBar()
        self.status.setStyleSheet(
            f"background: {PANEL}; color: {MUTED}; font-size: 11px;")
        self.setStatusBar(self.status)
        self.status.showMessage("Welcome! Click 'Select Images' to begin.")
        QApplication.instance().processEvents()
        self._reposition_kp_overlay()

    # ── HELPERS ─────────────────────────────────────────────────────────────

    def _reposition_kp_overlay(self, event=None):
        if hasattr(self, 'kp_overlay') and hasattr(self, 'canvas'):
            margin = 12
            ow = self.kp_overlay.width()
            oh = self.kp_overlay.height()
            cw = self.canvas.parent().width()
            ch = self.canvas.parent().height()
            self.kp_overlay.move(cw - ow - margin, ch - oh - margin)
        if event is not None:
            QWidget.resizeEvent(self.canvas.parent(), event)

    def _on_load_bbox_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Bbox Detection Model (ONNX)", "",
            "ONNX Model (*.onnx);;All Files (*)")
        if path:
            self._load_bbox_model(path)

    def _can_auto_save(self):
        kps = self.canvas.get_keypoints()
        return (
            self._save_crop_dir is not None
            and self._save_yolo_dir is not None
            and self._current_crop_image is not None
            and kps[0] is not None
            and kps[1] is not None
        )

    def _save_full_image(self):
        """
        Simpan gambar original (full/ori) ke subfolder full_images
        di dalam crop folder. Nama file sama dengan nama sumber asli.
        Dipanggil otomatis setiap kali crop disimpan.
        """
        if self._full_images_dir is None:
            return
        if self._image_bgr is None or self._current_idx < 0 or not self._image_paths:
            return

        src_path = self._image_paths[self._current_idx]
        fname    = os.path.basename(src_path)
        out_path = os.path.join(self._full_images_dir, fname)
        cv2.imwrite(out_path, self._image_bgr)
        print(f"[FULL] Saved original -> {out_path}")

    def _auto_save(self):
        if not self._can_auto_save():
            return

        src_path  = self._image_paths[self._current_idx]
        fname     = os.path.basename(src_path)
        stem, ext = os.path.splitext(fname)

        # Simpan crop
        cv2.imwrite(os.path.join(self._save_crop_dir, stem + ext),
                    self._current_crop_image)

        # Simpan original ke full_images
        self._save_full_image()

        # Simpan YOLO label + meta
        kps      = self.canvas.get_keypoints()
        img_w, img_h = self.canvas.get_image_size()
        yolo_out = os.path.join(self._save_yolo_dir, stem + ".txt")
        export_yolo_keypoints(kps, img_w, img_h, src_path, yolo_out,
                              crop_box=self._current_crop_box)

        full_info = f"  +  full_images/{fname}" if self._full_images_dir else ""
        self.status.showMessage(
            f"Auto-saved: {stem + ext}  +  {stem}.txt  +  {stem}_meta.json{full_info}")

    def _find_yolo_annotation_for_current_image(self):
        if self._current_idx < 0 or not self._image_paths:
            return None

        src_path = self._image_paths[self._current_idx]
        txt_name = os.path.splitext(os.path.basename(src_path))[0] + ".txt"

        candidate_dirs = []
        if self._annot_dir:
            candidate_dirs.append(self._annot_dir)
        candidate_dirs.append(os.path.dirname(src_path))
        if self._save_yolo_dir:
            candidate_dirs.append(self._save_yolo_dir)

        for directory in candidate_dirs:
            candidate = os.path.join(directory, txt_name)
            if os.path.isfile(candidate):
                return candidate

        return None

    def _load_yolo_from_path(self, path):
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        meta      = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                print(f"[META] Found meta.json: {meta_path}")
            except Exception as e:
                print(f"[META] Failed to read meta.json: {e}")

        img_w, img_h = self.canvas.get_image_size()
        kps_loaded   = [None, None]
        remapped     = False

        if meta is not None and self._current_crop_box is not None:
            x1, y1, x2, y2 = self._current_crop_box
            new_w = x2 - x1
            new_h = y2 - y1

            for kp_meta in meta.get("keypoints", []):
                i    = kp_meta["id"] - 1
                if i < 0 or i >= 2:
                    continue
                orig = kp_meta.get("original_absolute")
                if orig is None:
                    continue

                ox, oy    = orig
                new_x_rel = (ox - x1) / new_w
                new_y_rel = (oy - y1) / new_h

                if 0.0 <= new_x_rel <= 1.0 and 0.0 <= new_y_rel <= 1.0:
                    kps_loaded[i] = (int(round(new_x_rel * img_w)),
                                     int(round(new_y_rel * img_h)))
                else:
                    print(f"[META] KP{i+1} out of new bbox after remap "
                          f"(orig={orig}, rel=({new_x_rel:.3f},{new_y_rel:.3f}))")

            remapped = True

        else:
            with open(path, "r") as f:
                line = f.readline().strip()

            parts = line.split()
            if len(parts) < 11:
                raise ValueError(
                    "YOLO file does not have enough keypoint fields.\n"
                    "At least 11 columns are required.")

            for i in range(2):
                base   = 5 + i * 3
                x_norm = float(parts[base])
                y_norm = float(parts[base + 1])
                vis    = int(parts[base + 2])

                if vis == 0 or (x_norm == 0.0 and y_norm == 0.0):
                    continue

                kps_loaded[i] = (int(round(x_norm * img_w)),
                                 int(round(y_norm * img_h)))

        contour_pts = self.canvas._contour_pts
        if contour_pts:
            pts_arr = np.array(contour_pts, dtype=float)
            for i, kp in enumerate(kps_loaded):
                if kp is not None:
                    dists = np.hypot(pts_arr[:, 0] - kp[0], pts_arr[:, 1] - kp[1])
                    kps_loaded[i] = contour_pts[int(np.argmin(dists))]

        self.canvas.load_keypoints(kps_loaded)
        n_loaded = sum(1 for k in kps_loaded if k is not None)

        self.btn_reset_kp1.setEnabled(kps_loaded[0] is not None)
        self.btn_reset_kp2.setEnabled(kps_loaded[1] is not None)

        return n_loaded, remapped

    def _goto(self, idx: int, auto_save_current: bool = False):
        if not self._image_paths or not (0 <= idx < len(self._image_paths)):
            return

        if auto_save_current and self._current_idx >= 0:
            self._auto_save()

        self._current_idx = idx
        path = self._image_paths[idx]
        img  = cv2.imread(path)
        if img is None:
            self.status.showMessage(f"Failed to read: {path}")
            return
        self._image_bgr = img
        h, w = img.shape[:2]

        self.canvas.set_image_only(img)
        self._current_crop_image = None
        self._current_crop_box   = None

        n = len(self._image_paths)
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < n - 1)
        self.lbl_counter.setText(f"{idx+1} / {n}")

        fname_display = os.path.basename(path)
        if len(fname_display) > 26:
            fname_display = fname_display[:12] + "…" + fname_display[-11:]
        self.lbl_filename.setText(fname_display)

        self.btn_segment.setEnabled(True)
        self.btn_toggle_overlay.setEnabled(False)
        self.btn_toggle_kp.setEnabled(False)
        self.btn_toggle_context.setEnabled(False)
        self.btn_toggle_context.setText("🔭  Context View: OFF")
        self.btn_reset_kp1.setEnabled(False)
        self.btn_reset_kp2.setEnabled(False)
        self.btn_export_yolo.setEnabled(False)
        self.btn_save_crop.setEnabled(False)
        self.btn_load_yolo.setEnabled(False)
        for lbl in self.kp_labels:
            lbl.setText(lbl.text().split(":")[0] + ": -")

        self.status.showMessage(
            f"[{idx+1}/{n}] {os.path.basename(path)}  ({w}x{h})"
            " — Click 'Segment + Crop Neck'.")

    def _set_status(self, msg):
        self.status.showMessage(msg)

    def _on_kp_changed(self):
        kps = self.canvas.get_keypoints()
        for i, kp in enumerate(kps):
            self.kp_labels[i].setText(
                f"KP{i+1}: ({kp[0]}, {kp[1]})" if kp else f"KP{i+1}: -")
        has_kp    = any(k is not None for k in kps)
        segmented = bool(self.canvas._contour_pts) or self.canvas._showing_original
        self.btn_export_yolo.setEnabled(has_kp)
        self.btn_reset_kp1.setEnabled(segmented and kps[0] is not None)
        self.btn_reset_kp2.setEnabled(segmented and kps[1] is not None)
        self.kp_overlay.adjustSize()
        self._reposition_kp_overlay()

    def _on_load(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images (multiple allowed)", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not paths:
            return
        self._image_paths = sorted(paths)
        self.lbl_filename.setText("–")
        self._goto(0)

    def _on_prev(self):
        self._goto(self._current_idx - 1, auto_save_current=True)

    def _on_next(self):
        self._goto(self._current_idx + 1, auto_save_current=True)

    def _on_segment(self):
        if self._image_bgr is None:
            return

        if self._bbox_detector is None:
            QMessageBox.warning(self, "No Bbox Model",
                                "ONNX bbox model not loaded.\n"
                                "Use 'Change Bbox Model (.onnx)' to load a .onnx file.")
            return

        self.status.showMessage("Segmenting...")
        QApplication.processEvents()

        mask, _, _ = self._segmenter.segment(self._image_bgr)
        all_contour_pts = extract_contour_points(mask, max_points=600)

        if not all_contour_pts:
            QMessageBox.warning(self, "Segmentation", "No contour found.")
            return

        self.status.showMessage("Detecting neck area (ONNX)...")
        QApplication.processEvents()

        cropped_img, crop_box = crop_to_bbox(
            self._image_bgr, self._bbox_detector, padding=10)

        print(f"[GUI] image shape: {self._image_bgr.shape}")
        print(f"[GUI] crop_box: {crop_box}")

        if cropped_img is None or cropped_img.size == 0:
            QMessageBox.warning(self, "Neck Detection",
                                "Failed to detect neck area.")
            self.canvas.load(self._image_bgr, all_contour_pts)
            return

        cropped_contour_pts      = filter_and_offset_contour(all_contour_pts, crop_box)
        self._current_crop_image = cropped_img.copy()
        self._current_crop_box   = crop_box

        self.canvas.load(cropped_img, cropped_contour_pts if cropped_contour_pts else [])
        self.canvas.set_context(cropped_img, self._image_bgr, crop_box)

        self.btn_toggle_overlay.setEnabled(True)
        self.btn_toggle_kp.setEnabled(True)
        self.btn_toggle_context.setEnabled(True)
        self.btn_save_crop.setEnabled(True)
        self.btn_load_yolo.setEnabled(True)

        self.status.showMessage(
            f"Done: {len(cropped_contour_pts)} contour points in neck area.")

        self._try_auto_load_yolo_annotation()

    def _try_auto_load_yolo_annotation(self):
        annotation_path = self._find_yolo_annotation_for_current_image()
        if annotation_path is None:
            return
        try:
            n_loaded, remapped = self._load_yolo_from_path(annotation_path)
            mode = " [remapped via meta.json]" if remapped else ""
            self.status.showMessage(
                f"Auto-loaded annotation ({n_loaded} KP) from: "
                f"{os.path.basename(annotation_path)}{mode}")
        except Exception as e:
            self.status.showMessage(
                f"Found annotation '{os.path.basename(annotation_path)}'"
                f" but failed to parse it: {e}")

    def _on_toggle_overlay(self):
        self.canvas.toggle_overlay()
        vis = self.canvas._overlay_visible
        self.btn_toggle_overlay.setText(f"👁  Overlay: {'ON' if vis else 'OFF'}")

    def _on_toggle_kp(self):
        self.canvas.toggle_keypoints()
        vis = self.canvas._keypoints_visible
        self.btn_toggle_kp.setText(f"📍  Keypoints: {'ON' if vis else 'OFF'}")

    def _on_toggle_context(self):
        if not self.canvas.can_toggle_context():
            return
        success = self.canvas.toggle_context_view()
        if success:
            is_orig = self.canvas.is_showing_original()
            self.btn_toggle_context.setText(
                f"🔭  Context View: {'ON' if is_orig else 'OFF'}")
            self.btn_reset_kp1.setEnabled(
                not is_orig and self.canvas.get_keypoints()[0] is not None)
            self.btn_reset_kp2.setEnabled(
                not is_orig and self.canvas.get_keypoints()[1] is not None)
            if is_orig:
                self.status.showMessage(
                    "CONTEXT MODE: Full image view. Orange box = crop area."
                    " Click 'Context View: ON' to return to annotation.")
            else:
                self.status.showMessage(
                    "Back to crop mode. Click to add/change keypoints.")

    def _on_toggle_snap(self):
        new_val = not self.canvas._snap_to_contour
        self.canvas.set_snap_to_contour(new_val)
        self.btn_toggle_snap.setText(
            f"🧲  Snap to Contour: {'ON' if new_val else 'OFF'}")
        self.canvas.update()
        self.status.showMessage(
            "Snap to contour ON — clicks snap to nearest contour point."
            if new_val else
            "Snap to contour OFF — clicks place keypoints anywhere on the image.")

    def _on_reset_kp(self, idx: int):
        self.canvas.reset_kp(idx)

    def _on_set_annot_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Annotation + Meta JSON Folder")
        if folder:
            self._annot_dir = folder
            display = folder if len(folder) <= 28 else "..." + folder[-25:]
            self.lbl_annot_dir.setText(f"Folder: {display}")
            self.status.showMessage(
                f"Annotation folder set: {folder} — auto-load when filename matches.")

    def _on_set_crop_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Crops")
        if folder:
            self._save_crop_dir = folder

            # ── Buat subfolder full_images otomatis ──────────────────────
            self._full_images_dir = os.path.join(folder, "full_images")
            os.makedirs(self._full_images_dir, exist_ok=True)

            display = folder if len(folder) <= 28 else "..." + folder[-25:]
            self.lbl_crop_dir.setText(f"Folder: {display}")
            self.lbl_full_images_dir.setText("📂 full_images/ subfolder ready")
            self.status.showMessage(
                f"Crop folder set: {folder}  |  full_images/ subfolder created.")

    def _on_save_crop(self):
        if self._current_crop_image is None:
            QMessageBox.warning(self, "Save Crop",
                                "No crop image available. Run segmentation first.")
            return
        if self._current_idx < 0:
            return

        src_path  = self._image_paths[self._current_idx]
        fname     = os.path.basename(src_path)
        stem, ext = os.path.splitext(fname)
        save_name = stem + "_crop" + ext

        if self._save_crop_dir:
            out_path = os.path.join(self._save_crop_dir, save_name)
            cv2.imwrite(out_path, self._current_crop_image)

            # ── Simpan juga gambar original ke full_images ───────────────
            self._save_full_image()

            full_info = f"\nOriginal saved to: {self._full_images_dir}" \
                        if self._full_images_dir else ""
            self.status.showMessage(f"Crop saved: {out_path}{' + full_images/' + fname if self._full_images_dir else ''}")
            QMessageBox.information(self, "Crop Saved",
                                    f"Crop image saved to:\n{out_path}{full_info}")
        else:
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Save Crop Image", save_name,
                "Images (*.png *.jpg *.jpeg)")
            if not out_path:
                return
            cv2.imwrite(out_path, self._current_crop_image)

            # ── Simpan juga gambar original ke full_images (jika ada) ────
            self._save_full_image()

            full_info = f"\nOriginal saved to: {self._full_images_dir}" \
                        if self._full_images_dir else ""
            self.status.showMessage(f"Crop saved: {out_path}")
            QMessageBox.information(self, "Crop Saved",
                                    f"Crop image saved to:\n{out_path}{full_info}")

    def _on_set_yolo_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save YOLO Labels")
        if folder:
            self._save_yolo_dir = folder
            display = folder if len(folder) <= 28 else "..." + folder[-25:]
            self.lbl_yolo_dir.setText(f"Folder: {display}")
            self.status.showMessage(f"YOLO folder set: {folder}")

    def _on_export_yolo(self):
        kps = self.canvas.get_keypoints()
        sz  = self.canvas.get_image_size()
        if sz is None or self._current_idx < 0:
            return
        img_w, img_h = sz
        path         = self._image_paths[self._current_idx]
        stem         = os.path.splitext(os.path.basename(path))[0]
        default_name = stem + ".txt"

        if self._save_yolo_dir:
            out_path = os.path.join(self._save_yolo_dir, default_name)
            export_yolo_keypoints(kps, img_w, img_h, path, out_path,
                                  crop_box=self._current_crop_box)
            self.status.showMessage(
                f"YOLO label saved: {out_path}  +  {stem}_meta.json")
            QMessageBox.information(
                self, "Export Successful",
                f"YOLO label saved to:\n{out_path}\n\n"
                f"Metadata saved to:\n"
                f"{os.path.join(self._save_yolo_dir, stem + '_meta.json')}")
        else:
            out, _ = QFileDialog.getSaveFileName(
                self, "Save YOLO Label", default_name, "YOLO Label (*.txt)")
            if not out:
                return
            export_yolo_keypoints(kps, img_w, img_h, path, out,
                                  crop_box=self._current_crop_box)
            self.status.showMessage(f"YOLO label saved: {out}")
            QMessageBox.information(
                self, "Export Successful",
                f"YOLO label saved to:\n{out}\n\n"
                f"Metadata saved to:\n{os.path.splitext(out)[0] + '_meta.json'}")

    def _on_load_yolo_annotation(self):
        if self._current_crop_image is None:
            QMessageBox.warning(self, "Load Annotation",
                                "Run segmentation first so contour is available.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Annotation File", "",
            "YOLO Label (*.txt)")
        if not path:
            return

        try:
            n_loaded, remapped = self._load_yolo_from_path(path)
            meta_path = os.path.splitext(path)[0] + "_meta.json"
            if remapped:
                self.status.showMessage(
                    f"Loaded {n_loaded} KP from {os.path.basename(path)}"
                    f" [remapped via {os.path.basename(meta_path)}]."
                    " Verify and export when ready.")
            else:
                self.status.showMessage(
                    f"Loaded {n_loaded} KP from {os.path.basename(path)}"
                    " [standard mode, no meta.json found]."
                    " Verify and export when ready.")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load annotation:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())