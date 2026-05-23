import sys
import os
import cv2
import numpy as np
from pathlib import Path
from config_bbox import MODEL_DIR

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QFrame, QMessageBox,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem,
    QListWidget, QListWidgetItem, QScrollArea, QDialog
)
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont, QCursor
)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


DARK     = "#12121a"
PANEL    = "#1e1e2e"
ACCENT   = "#4f8ef7"
MUTED    = "#64748b"
BTN_TEXT = "#e2e8f0"
SUCCESS  = "#22c55e"
WARN     = "#f59e0b"
DANGER   = "#ef4444"
ACCENT2  = "#60a5fa"
CARD     = "#0d0d16"

DEFAULT_BBOX_MODEL = MODEL_DIR / "good_augwed/weights/best.onnx"


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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class NeckDetector:
    def __init__(self, model_path: str):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")
        self.onnx = ort.InferenceSession(model_path)

    def detect(self, image: np.ndarray, padding: int = 0):
        h_orig, w_orig = image.shape[:2]
        resized = cv2.resize(image, (640, 640))
        inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]

        output_raw = self.onnx.run(None, {self.onnx.get_inputs()[0].name: inp})[0]
        output = output_raw[0].T

        if len(output) == 0:
            return []

        candidates = []
        sx = w_orig / 640.0
        sy = h_orig / 640.0

        for row in output:
            cx, cy, w, h, score = row[:5]
            if score < 0.1:
                continue
            x1 = max(0, int(cx * sx - (w * sx) / 2) - padding)
            y1 = max(0, int(cy * sy - (h * sy) / 2) - padding)
            x2 = min(w_orig, int(cx * sx + (w * sx) / 2) + padding)
            y2 = min(h_orig, int(cy * sy + (h * sy) / 2) + padding)
            candidates.append((x1, y1, x2, y2, float(score)))

        if not candidates:
            return []

        candidates.sort(key=lambda d: d[4], reverse=True)
        kept = []
        for det in candidates:
            suppress = False
            for k in kept:
                if iou(det[:4], k[:4]) > 0.5:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)

        best = max(kept, key=lambda d: d[4])
        return [best]


class BBox:
    def __init__(self, x1, y1, x2, y2, score=1.0, class_id=0):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.score = score
        self.class_id = class_id
        self.selected = False

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1

    def to_yolo(self, img_w, img_h):
        return f"{self.class_id} {self.cx/img_w:.6f} {self.cy/img_h:.6f} {self.w/img_w:.6f} {self.h/img_h:.6f}"

    def to_yolo_normalized(self, img_w, img_h):
        return (self.class_id, self.cx/img_w, self.cy/img_h, self.w/img_w, self.h/img_h)

    @classmethod
    def from_yolo_normalized(cls, class_id, cx_n, cy_n, w_n, h_n, img_w, img_h, score=1.0):
        x1 = int((cx_n - w_n / 2) * img_w)
        y1 = int((cy_n - h_n / 2) * img_h)
        x2 = int((cx_n + w_n / 2) * img_w)
        y2 = int((cy_n + h_n / 2) * img_h)
        return cls(x1, y1, x2, y2, score=score, class_id=class_id)

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def handle_at(self, x, y, tol=10):
        corners = {
            'tl': (self.x1, self.y1), 'tr': (self.x2, self.y1),
            'bl': (self.x1, self.y2), 'br': (self.x2, self.y2),
        }
        mids = {
            'ml': (self.x1, self.cy), 'mr': (self.x2, self.cy),
            'mt': (self.cx, self.y1), 'mb': (self.cx, self.y2),
        }
        for name, (hx, hy) in {**corners, **mids}.items():
            if abs(x - hx) <= tol and abs(y - hy) <= tol:
                return name
        return None

    def move(self, dx, dy, img_w, img_h):
        self.x1 = max(0, min(img_w - 1, self.x1 + dx))
        self.y1 = max(0, min(img_h - 1, self.y1 + dy))
        self.x2 = max(0, min(img_w - 1, self.x2 + dx))
        self.y2 = max(0, min(img_h - 1, self.y2 + dy))

    def resize_handle(self, handle, dx, dy, img_w, img_h):
        if 'l' in handle: self.x1 = max(0, min(self.x2 - 5, self.x1 + dx))
        if 'r' in handle: self.x2 = max(self.x1 + 5, min(img_w, self.x2 + dx))
        if 't' in handle: self.y1 = max(0, min(self.y2 - 5, self.y1 + dy))
        if 'b' in handle: self.y2 = max(self.y1 + 5, min(img_h, self.y2 + dy))

    def normalize(self):
        if self.x1 > self.x2: self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2: self.y1, self.y2 = self.y2, self.y1

    def clone(self):
        return BBox(self.x1, self.y1, self.x2, self.y2, self.score, self.class_id)


ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.1, 8.0, 1.15


class AnnotationCanvas(QWidget):
    status_message = pyqtSignal(str)
    bboxes_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.WheelFocus)

        self._base_image: np.ndarray | None = None
        self._bboxes: list[BBox] = []
        self._base_pix: QPixmap | None = None

        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._pan_start = None
        self._pan_origin = None

        self._drag_state: dict | None = None
        self._draw_state: dict | None = None

        self._threshold = 0.5

    def load(self, image_bgr: np.ndarray):
        self._base_image = image_bgr.copy()
        self._bboxes = []
        self._build_base_pix()
        self._fit_to_view()
        self.update()
        self.bboxes_changed.emit()
        self.status_message.emit("Image loaded. Draw a bbox or run Auto Detect.")

    def set_image_only(self, image_bgr: np.ndarray):
        self.load(image_bgr)

    def set_bboxes(self, bboxes: list[BBox]):
        self._bboxes = bboxes
        self.update()
        self.bboxes_changed.emit()

    def get_bboxes(self) -> list[BBox]:
        return self._bboxes

    def get_image_size(self):
        if self._base_image is None:
            return None
        h, w = self._base_image.shape[:2]
        return w, h

    def set_threshold(self, v: float):
        self._threshold = v
        self.update()
        self.bboxes_changed.emit()

    def clear_bboxes(self):
        self._bboxes = []
        self.update()
        self.bboxes_changed.emit()

    def delete_selected(self):
        self._bboxes = [b for b in self._bboxes if not b.selected]
        self.update()
        self.bboxes_changed.emit()

    def zoom_in(self):
        self._apply_zoom(ZOOM_STEP, self._center())

    def zoom_out(self):
        self._apply_zoom(1 / ZOOM_STEP, self._center())

    def zoom_reset(self):
        self._fit_to_view()
        self.update()

    def _center(self):
        return QPointF(self.width() / 2, self.height() / 2)

    def _fit_to_view(self):
        if self._base_pix is None:
            return
        pw, ph = self._base_pix.width(), self._base_pix.height()
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        self._zoom = min(ww / pw, wh / ph)
        self._pan = QPointF((ww - pw * self._zoom) / 2,
                            (wh - ph * self._zoom) / 2)

    def _apply_zoom(self, factor, pivot: QPointF):
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

    def _image_to_widget(self, ix, iy):
        return (ix * self._zoom + self._pan.x(),
                iy * self._zoom + self._pan.y())

    def _build_base_pix(self):
        if self._base_image is None:
            self._base_pix = None
            return
        img = self._base_image
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self._base_pix = QPixmap.fromImage(qimg)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 24))

        if self._base_pix is None:
            painter.setPen(QColor(100, 100, 120))
            painter.setFont(QFont("Consolas", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No image loaded.\nClick 'Open Images'.")
            return

        pw = self._base_pix.width() * self._zoom
        ph = self._base_pix.height() * self._zoom
        target = QRectF(self._pan.x(), self._pan.y(), pw, ph)
        painter.drawPixmap(target, self._base_pix, QRectF(self._base_pix.rect()))

        painter.setPen(QPen(QColor("#1e2d45"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target)

        painter.setRenderHint(QPainter.Antialiasing)
        for bbox in self._bboxes:
            self._paint_bbox(painter, bbox)

        if self._draw_state and self._draw_state.get("active"):
            x0, y0 = self._image_to_widget(self._draw_state["x0"], self._draw_state["y0"])
            x1, y1 = self._draw_state["cur_wx"], self._draw_state["cur_wy"]
            pen = QPen(QColor(ACCENT2), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())

        painter.setPen(QColor(160, 160, 180))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(8, self.height() - 8, f"zoom {self._zoom:.2f}x")

    def _paint_bbox(self, painter: QPainter, bbox: BBox):
        wx1, wy1 = self._image_to_widget(bbox.x1, bbox.y1)
        wx2, wy2 = self._image_to_widget(bbox.x2, bbox.y2)
        rect = QRectF(QPointF(wx1, wy1), QPointF(wx2, wy2))

        will_save = bbox.score >= self._threshold

        if bbox.selected:
            color = QColor(SUCCESS)
            width = 3
        elif not will_save:
            color = QColor(WARN)
            width = 2
        else:
            color = QColor(ACCENT2)
            width = 2

        if bbox.selected:
            fill = QColor(SUCCESS)
            fill.setAlpha(30)
            painter.fillRect(rect, fill)

        pen = QPen(color, width)
        if not will_save and not bbox.selected:
            pen.setStyle(Qt.DashLine)
        else:
            pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        save_icon = "✗" if not will_save else "✓"
        label_text = f" {save_icon} {bbox.score:.2f} "
        score_color = QColor(DANGER) if not will_save else QColor(BTN_TEXT)
        bg_rect = QRectF(wx1 + 2, wy1 - 16, len(label_text) * 6.5, 14)
        painter.fillRect(bg_rect, QColor(18, 18, 24))
        painter.setPen(score_color)
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.drawText(QPointF(wx1 + 4, wy1 - 4), label_text)

        if bbox.selected:
            handles = [
                (bbox.x1, bbox.y1), (bbox.x2, bbox.y1),
                (bbox.x1, bbox.y2), (bbox.x2, bbox.y2),
                (bbox.cx, bbox.y1), (bbox.cx, bbox.y2),
                (bbox.x1, bbox.cy), (bbox.x2, bbox.cy),
            ]
            for hx, hy in handles:
                hwx, hwy = self._image_to_widget(hx, hy)
                painter.setPen(QPen(QColor(BTN_TEXT), 1))
                painter.setBrush(QBrush(QColor(ACCENT)))
                painter.drawRect(QRectF(hwx - 5, hwy - 5, 10, 10))

    def wheelEvent(self, event):
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self._apply_zoom(factor, QPointF(event.pos()))

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_start = event.pos()
            self._pan_origin = QPointF(self._pan)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return
        if event.button() == Qt.LeftButton:
            self._handle_left_press(event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan = self._pan_origin + QPointF(delta)
            self.update()
            return
        if self._drag_state:
            self._handle_drag_move(event.x(), event.y())
        elif self._draw_state and self._draw_state.get("active"):
            self._draw_state["cur_wx"] = event.x()
            self._draw_state["cur_wy"] = event.y()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_start = None
            self._pan_origin = None
            self.setCursor(QCursor(Qt.CrossCursor))
            return
        if event.button() == Qt.LeftButton:
            self._handle_left_release(event.x(), event.y())

    def resizeEvent(self, event):
        if self._base_pix is not None:
            self._fit_to_view()
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_selected()

    def _handle_left_press(self, wx, wy):
        if self._base_image is None:
            return
        ix, iy = self._widget_to_image(wx, wy)
        tol = 8 / self._zoom

        for b in self._bboxes:
            if not b.selected:
                continue
            handle = b.handle_at(ix, iy, tol=tol)
            if handle:
                self._drag_state = {
                    "mode": "resize", "bbox": b,
                    "handle": handle, "last_ix": ix, "last_iy": iy,
                }
                return

        hit = None
        for b in reversed(self._bboxes):
            if b.contains(ix, iy):
                hit = b
                break

        if hit:
            for b in self._bboxes:
                b.selected = False
            hit.selected = True
            self._drag_state = {
                "mode": "move", "bbox": hit,
                "last_ix": ix, "last_iy": iy,
            }
        else:
            for b in self._bboxes:
                b.selected = False
            self._draw_state = {
                "active": True,
                "x0": ix, "y0": iy,
                "cur_wx": wx, "cur_wy": wy,
            }

        self.update()
        self.bboxes_changed.emit()

    def _handle_drag_move(self, wx, wy):
        if self._base_image is None or not self._drag_state:
            return
        ix, iy = self._widget_to_image(wx, wy)
        ih, iw = self._base_image.shape[:2]
        b = self._drag_state["bbox"]
        dx = int(ix - self._drag_state["last_ix"])
        dy = int(iy - self._drag_state["last_iy"])

        if self._drag_state["mode"] == "move":
            b.move(dx, dy, iw, ih)
        elif self._drag_state["mode"] == "resize":
            b.resize_handle(self._drag_state["handle"], dx, dy, iw, ih)

        self._drag_state["last_ix"] = ix
        self._drag_state["last_iy"] = iy
        self.update()

    def _handle_left_release(self, wx, wy):
        if self._drag_state:
            self._drag_state["bbox"].normalize()
            self._drag_state = None
            self.update()
            self.bboxes_changed.emit()
            return

        if self._draw_state and self._draw_state.get("active"):
            if self._base_image is None:
                self._draw_state = None
                return
            ih, iw = self._base_image.shape[:2]
            ix, iy = self._widget_to_image(wx, wy)
            x0, y0 = self._draw_state["x0"], self._draw_state["y0"]
            x1 = max(0, min(iw, int(min(x0, ix))))
            y1 = max(0, min(ih, int(min(y0, iy))))
            x2 = max(0, min(iw, int(max(x0, ix))))
            y2 = max(0, min(ih, int(max(y0, iy))))
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                new_bbox = BBox(x1, y1, x2, y2, score=1.0)
                new_bbox.selected = True
                for b in self._bboxes:
                    b.selected = False
                self._bboxes.append(new_bbox)
                self.bboxes_changed.emit()
            self._draw_state = None
            self.update()


# ─────────────────────────────────────────────
#  Crop Preview Dialog
# ─────────────────────────────────────────────
class CropDialog(QDialog):
    def __init__(self, crop_img: np.ndarray, bbox: BBox, threshold: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Crop Preview  [{bbox.x1},{bbox.y1} → {bbox.x2},{bbox.y2}]")
        self.setStyleSheet(f"background: {DARK}; color: {BTN_TEXT};")
        self.setMinimumSize(400, 300)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)

        ch, cw = crop_img.shape[:2]
        will_save = bbox.score >= threshold
        info_color = DANGER if not will_save else ACCENT2
        save_text = "⚠  Will NOT be saved (low-conf)" if not will_save else "✓  Will be saved"

        info = QLabel(f"Original: {cw} × {ch} px     Confidence: {bbox.score:.3f}")
        info.setStyleSheet(f"color: {info_color}; font-size: 11px; font-family: Consolas; font-weight: bold;")
        info.setAlignment(Qt.AlignCenter)
        lay.addWidget(info)

        save_lbl = QLabel(save_text)
        save_lbl.setStyleSheet(f"color: {WARN if not will_save else SUCCESS}; font-size: 10px; font-family: Consolas;")
        save_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(save_lbl)

        scale = min(600 / max(cw, 1), 500 / max(ch, 1), 1.0)
        dw = max(1, int(cw * scale))
        dh = max(1, int(ch * scale))
        resized = cv2.resize(crop_img, (dw, dh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, dw, dh, rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        img_lbl = QLabel()
        img_lbl.setPixmap(pix)
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setStyleSheet(f"border: 1px solid #2d2d3d;")
        lay.addWidget(img_lbl)

        close_btn = _btn("  Close", DANGER, min_w=100, h=34)
        close_btn.clicked.connect(self.accept)
        lay.addWidget(close_btn, alignment=Qt.AlignCenter)


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    LOW_CONF_THRESHOLD = 0.5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO BBox Annotator")
        self.resize(1700, 900)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(f"background: {DARK}; color: #e2e8f0;")

        self._detector: NeckDetector | None = None
        self._image_paths: list[Path] = []
        self._current_idx = -1
        self._image_bgr: np.ndarray | None = None
        self._save_dir: Path | None = None
        self._threshold = self.LOW_CONF_THRESHOLD

        self._low_conf_log: dict[int, dict] = {}
        self._clipboard: dict | None = None

        self._build_ui()

        # ── Auto-load default model (sama seperti pola annotator keypoint) ──
        self._load_model(str(DEFAULT_BBOX_MODEL))

    # ─────────────────────────────────────────
    #  Model Loading (dipisah jadi helper + slot)
    # ─────────────────────────────────────────

    def _load_model(self, path: str):
        """
        Load ONNX model dari path. Dipanggil saat startup (default) maupun
        saat user memilih file baru via dialog.
        """
        if not path or not os.path.isfile(path):
            self.lbl_model.setText("Model: not found — use Change Model")
            self.lbl_model.setStyleSheet(
                f"color: {WARN}; font-size: 9px; font-family: Consolas;")
            return

        if not ONNX_AVAILABLE:
            self.lbl_model.setText("onnxruntime not installed!")
            self.lbl_model.setStyleSheet(
                f"color: {DANGER}; font-size: 9px; font-family: Consolas;")
            return

        try:
            self._detector = NeckDetector(path)
            short = Path(path).name
            self.lbl_model.setText(f"Active: {short}")
            self.lbl_model.setStyleSheet(
                f"color: {SUCCESS}; font-size: 9px; font-family: Consolas;")
            self._set_status(f"ONNX model loaded: {short}")
        except Exception as e:
            self._detector = None
            self.lbl_model.setText("Model: load failed — use Change Model")
            self.lbl_model.setStyleSheet(
                f"color: {DANGER}; font-size: 9px; font-family: Consolas;")
            self._set_status(f"Model load error: {e}")

    def _on_load_model(self):
        """Buka file dialog lalu panggil _load_model."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", "",
            "ONNX Model (*.onnx);;All Files (*.*)")
        if path:
            self._load_model(path)

    # ─────────────────────────────────────────
    #  UI Builder
    # ─────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left Sidebar ─────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(
            f"background: {PANEL}; border-right: 1px solid #2d2d3d;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(12, 18, 12, 12)
        sb.setSpacing(7)

        title = QLabel("▣  BBOX\nAnnotator")
        title.setStyleSheet(
            f"color: {ACCENT}; font-family: Consolas; "
            "font-size: 18px; font-weight: 700; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignCenter)
        sb.addWidget(title)
        sb.addWidget(_divider())

        # ── MODEL ────────────────────────────────────────────────────────────
        sb.addWidget(_section("BBOX MODEL"))

        # Label status model (sama persis dengan annotator keypoint)
        self.lbl_model = QLabel("Model: initializing...")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        sb.addWidget(self.lbl_model)

        # Tombol ganti model (bukan "Load" tapi "Change", default sudah di-set)
        self.btn_load_model = _btn("📦  Change ONNX Model", "#6d28d9")
        self.btn_load_model.clicked.connect(self._on_load_model)
        sb.addWidget(self.btn_load_model)

        pad_row = QHBoxLayout()
        pad_row.setSpacing(6)
        pad_lbl = QLabel("Padding (px):")
        pad_lbl.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        self.spin_padding = QSpinBox()
        self.spin_padding.setRange(0, 200)
        self.spin_padding.setValue(0)
        self.spin_padding.setFixedWidth(60)
        self.spin_padding.setStyleSheet(
            f"background: {CARD}; color: {BTN_TEXT}; border: 1px solid #2d2d3d; "
            "border-radius: 4px; font-family: Consolas; font-size: 10px; padding: 2px;")
        pad_row.addWidget(pad_lbl)
        pad_row.addWidget(self.spin_padding)
        sb.addLayout(pad_row)

        sb.addWidget(_divider())

        # IMAGES
        sb.addWidget(_section("IMAGES"))
        self.btn_load_images = _btn("🖼  Open Images", ACCENT)
        self.btn_load_images.clicked.connect(self._on_load_images)
        sb.addWidget(self.btn_load_images)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self.btn_prev = _btn("◀ Prev", "#334155", 88)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(lambda: self._navigate(-1))
        self.btn_next = _btn("Next ▶", "#334155", 88)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(lambda: self._navigate(1))
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
            f"color: #e2e8f0; font-size: 11px; font-family: Consolas;"
            f"background: {CARD}; border: 1px solid #2d2d3d;"
            "border-radius: 8px; padding: 6px 8px;")
        sb.addWidget(self.lbl_filename)

        sb.addWidget(_divider())

        # SAVE
        sb.addWidget(_section("SAVE LABELS"))
        self.btn_set_save_dir = _btn("📁  Set Save Folder", "#0f766e")
        self.btn_set_save_dir.clicked.connect(self._on_set_save_dir)
        sb.addWidget(self.btn_set_save_dir)

        self.lbl_save_dir = QLabel("Folder: not set")
        self.lbl_save_dir.setWordWrap(True)
        self.lbl_save_dir.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        sb.addWidget(self.lbl_save_dir)

        self.btn_save_current = _btn("💾  Save Current", "#16a34a")
        self.btn_save_current.setEnabled(False)
        self.btn_save_current.clicked.connect(self._on_save_current)
        sb.addWidget(self.btn_save_current)

        note_lbl = QLabel("⚠ Only bboxes ≥ threshold are saved")
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(
            f"color: {WARN}; font-size: 9px; font-family: Consolas;"
            f"background: {CARD}; border: 1px solid #2d2d3d;"
            "border-radius: 4px; padding: 4px 6px;")
        sb.addWidget(note_lbl)

        sb.addWidget(_divider())

        # ACTIONS
        sb.addWidget(_section("ACTIONS"))
        self.btn_detect = _btn("🔍  Auto Detect (Best Conf)", "#0ea5e9")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._on_run_detection)
        sb.addWidget(self.btn_detect)

        self.btn_clear = _btn("🗑  Clear All BBoxes", "#7f1d1d")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self._on_clear_bboxes)
        sb.addWidget(self.btn_clear)

        self.btn_crop = _btn("✂  Crop Selected", "#7c3aed")
        self.btn_crop.setEnabled(False)
        self.btn_crop.clicked.connect(self._on_crop_selected)
        sb.addWidget(self.btn_crop)

        sb.addWidget(_divider())

        # ZOOM
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

        # CLIPBOARD
        sb.addWidget(_section("CLIPBOARD"))
        self.lbl_clipboard = QLabel("Clipboard empty")
        self.lbl_clipboard.setWordWrap(True)
        self.lbl_clipboard.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-family: Consolas;"
            f"background: {CARD}; border: 1px solid #2d2d3d;"
            "border-radius: 4px; padding: 5px 7px;")
        sb.addWidget(self.lbl_clipboard)

        self.btn_copy = _btn("📋  Copy Selected BBox", "#334155")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self._on_copy_selected)
        sb.addWidget(self.btn_copy)

        self.btn_paste_current = _btn("📌  Paste to This Image", "#334155")
        self.btn_paste_current.setEnabled(False)
        self.btn_paste_current.clicked.connect(self._on_paste_to_current)
        sb.addWidget(self.btn_paste_current)

        self.btn_paste_next = _btn("📌  Paste to Next Image", "#334155")
        self.btn_paste_next.setEnabled(False)
        self.btn_paste_next.clicked.connect(self._on_paste_to_next)
        sb.addWidget(self.btn_paste_next)

        sb.addStretch()

        # Shortcuts hint card
        hint_frame = QFrame()
        hint_frame.setStyleSheet(
            f"QFrame {{ background: {CARD}; border-radius: 8px; border: 1px solid #2d2d3d; }}")
        hf_lay = QVBoxLayout(hint_frame)
        hf_lay.setContentsMargins(8, 6, 8, 6)
        hf_lay.setSpacing(2)
        hf_lay.addWidget(_section("SHORTCUTS"))
        shortcuts = [
            ("←  /  →",    "Navigate images"),
            ("Click+Drag",  "Draw new bbox"),
            ("Click bbox",  "Select"),
            ("Drag edge",   "Resize"),
            ("Drag inside", "Move"),
            ("Delete",      "Remove selected"),
        ]
        for key, desc in shortcuts:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            k = QLabel(key)
            k.setFixedWidth(78)
            k.setStyleSheet(f"color: {ACCENT2}; font-size: 8px; font-family: Consolas; font-weight: bold; border: none;")
            d = QLabel(desc)
            d.setStyleSheet(f"color: {MUTED}; font-size: 8px; font-family: Consolas; border: none;")
            row_l.addWidget(k)
            row_l.addWidget(d)
            hf_lay.addWidget(row_w)
        sb.addWidget(hint_frame)
        sb.addSpacing(6)

        root.addWidget(sidebar)

        # ── Center: canvas + bbox list ────────────────────────────────────────
        center = QWidget()
        center.setStyleSheet(f"background: {DARK};")
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(8, 8, 8, 8)
        center_lay.setSpacing(8)

        self.canvas = AnnotationCanvas()
        self.canvas.status_message.connect(self._set_status)
        self.canvas.bboxes_changed.connect(self._on_bboxes_changed)
        center_lay.addWidget(self.canvas, stretch=1)

        # BBox list
        list_frame = QFrame()
        list_frame.setFixedHeight(140)
        list_frame.setStyleSheet(
            f"QFrame {{ background: {PANEL}; border: 1px solid #2d2d3d; border-radius: 6px; }}")
        lf_lay = QVBoxLayout(list_frame)
        lf_lay.setContentsMargins(8, 6, 8, 6)
        lf_lay.setSpacing(4)

        list_hdr = QHBoxLayout()
        list_hdr_lbl = QLabel("BBOX LIST")
        list_hdr_lbl.setStyleSheet(f"color: {ACCENT2}; font-size: 9px; font-weight: bold; font-family: Consolas;")
        self.lbl_bbox_count = QLabel("0 boxes")
        self.lbl_bbox_count.setStyleSheet(f"color: {MUTED}; font-size: 9px; font-family: Consolas;")
        list_hdr.addWidget(list_hdr_lbl)
        list_hdr.addStretch()
        list_hdr.addWidget(self.lbl_bbox_count)
        lf_lay.addLayout(list_hdr)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["ID", "X1", "Y1", "X2", "Y2", "Confidence", "Status"])
        self.tree.setColumnWidth(0, 40)
        for i in range(1, 5):
            self.tree.setColumnWidth(i, 65)
        self.tree.setColumnWidth(5, 90)
        self.tree.setColumnWidth(6, 90)
        self.tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {CARD};
                color: {BTN_TEXT};
                border: none;
                font-family: Consolas;
                font-size: 9px;
            }}
            QTreeWidget::item:selected {{
                background: {ACCENT};
                color: white;
            }}
            QHeaderView::section {{
                background: #1a2235;
                color: {ACCENT2};
                font-family: Consolas;
                font-size: 9px;
                font-weight: bold;
                border: none;
                padding: 3px;
            }}
        """)
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        lf_lay.addWidget(self.tree)

        center_lay.addWidget(list_frame)
        root.addWidget(center, stretch=1)

        # ── Right Panel: Low Conf Log ─────────────────────────────────────────
        right_panel = QFrame()
        right_panel.setFixedWidth(280)
        right_panel.setStyleSheet(
            f"background: {PANEL}; border-left: 1px solid #2d2d3d;")
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(12, 18, 12, 12)
        rp.setSpacing(7)

        log_title = QLabel("LOW CONF  LOG")
        log_title.setStyleSheet(
            f"color: {WARN}; font-family: Consolas; "
            "font-size: 14px; font-weight: 700; letter-spacing: 2px;")
        log_title.setAlignment(Qt.AlignCenter)
        rp.addWidget(log_title)

        warn_line = QFrame()
        warn_line.setFrameShape(QFrame.HLine)
        warn_line.setStyleSheet(f"color: {WARN};")
        rp.addWidget(warn_line)

        thresh_frame = QFrame()
        thresh_frame.setStyleSheet(
            f"QFrame {{ background: {CARD}; border-radius: 6px; border: 1px solid #2d2d3d; }}")
        tf_lay = QVBoxLayout(thresh_frame)
        tf_lay.setContentsMargins(10, 8, 10, 8)
        tf_lay.setSpacing(4)

        thresh_row = QHBoxLayout()
        thresh_lbl = QLabel("Threshold")
        thresh_lbl.setStyleSheet(f"color: {MUTED}; font-size: 9px; font-weight: bold;")
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 1.0)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(self.LOW_CONF_THRESHOLD)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setFixedWidth(70)
        self.spin_threshold.setStyleSheet(
            f"background: {CARD}; color: {ACCENT2}; border: 1px solid {ACCENT}; "
            "border-radius: 4px; font-family: Consolas; font-size: 10px; font-weight: bold; padding: 2px;")
        self.spin_threshold.valueChanged.connect(self._on_threshold_change)
        thresh_row.addWidget(thresh_lbl)
        thresh_row.addStretch()
        thresh_row.addWidget(self.spin_threshold)
        tf_lay.addLayout(thresh_row)

        thresh_note = QLabel("Bboxes below threshold are logged and NOT saved")
        thresh_note.setWordWrap(True)
        thresh_note.setStyleSheet(f"color: {WARN}; font-size: 8px; font-family: Consolas;")
        tf_lay.addWidget(thresh_note)
        rp.addWidget(thresh_frame)

        self.lbl_log_count = QLabel("0 problematic images")
        self.lbl_log_count.setAlignment(Qt.AlignCenter)
        self.lbl_log_count.setStyleSheet(
            f"color: {WARN}; font-size: 10px; font-weight: bold; font-family: Consolas;")
        rp.addWidget(self.lbl_log_count)

        self.log_list = QListWidget()
        self.log_list.setStyleSheet(f"""
            QListWidget {{
                background: {CARD};
                color: {WARN};
                border: 1px solid {WARN};
                border-radius: 4px;
                font-family: Consolas;
                font-size: 9px;
            }}
            QListWidget::item:selected {{
                background: {WARN};
                color: {DARK};
            }}
        """)
        self.log_list.itemDoubleClicked.connect(self._on_log_click)
        rp.addWidget(self.log_list, stretch=1)

        log_hint = QLabel("Double-click to jump to image")
        log_hint.setStyleSheet(f"color: {MUTED}; font-size: 8px; font-family: Consolas;")
        log_hint.setAlignment(Qt.AlignCenter)
        rp.addWidget(log_hint)

        self.btn_clear_log = _btn("🗑  Clear Log", "#7f1d1d")
        self.btn_clear_log.clicked.connect(self._on_clear_log)
        rp.addWidget(self.btn_clear_log)

        root.addWidget(right_panel)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            f"background: {PANEL}; color: {MUTED}; font-size: 11px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome! Click 'Open Images' to begin.")

        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence(Qt.Key_Left),  self).activated.connect(lambda: self._navigate(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self._navigate(1))
        QShortcut(QKeySequence(Qt.Key_Delete),self).activated.connect(self.canvas.delete_selected)

    # ─────────────────────────────────────────
    #  Slots & Helpers
    # ─────────────────────────────────────────

    def _set_status(self, msg):
        self.status_bar.showMessage(msg)

    def _on_bboxes_changed(self):
        self._update_tree()
        self._check_low_confidence()
        bboxes = self.canvas.get_bboxes()
        has_sel = any(b.selected for b in bboxes)
        self.btn_copy.setEnabled(has_sel)
        self.btn_crop.setEnabled(has_sel)
        self.btn_clear.setEnabled(len(bboxes) > 0)
        self.btn_save_current.setEnabled(len(bboxes) > 0 and self._save_dir is not None)

    def _on_tree_select(self):
        sel_items = self.tree.selectedItems()
        sel_ids = {int(it.text(0)) for it in sel_items}
        bboxes = self.canvas.get_bboxes()
        for i, b in enumerate(bboxes):
            b.selected = i in sel_ids
        self.canvas.update()
        self.btn_copy.setEnabled(any(b.selected for b in bboxes))
        self.btn_crop.setEnabled(any(b.selected for b in bboxes))

    def _update_tree(self):
        self.tree.clear()
        bboxes = self.canvas.get_bboxes()
        threshold = self._threshold
        for i, b in enumerate(bboxes):
            will_save = b.score >= threshold
            status = "✓ saved" if will_save else "✗ skipped"
            item = QTreeWidgetItem([
                str(i),
                str(b.x1), str(b.y1), str(b.x2), str(b.y2),
                f"{b.score:.3f}", status
            ])
            if not will_save:
                for col in range(7):
                    item.setForeground(col, QColor(WARN))
            elif b.selected:
                for col in range(7):
                    item.setForeground(col, QColor(SUCCESS))
            self.tree.addTopLevelItem(item)
        n = len(bboxes)
        self.lbl_bbox_count.setText(f"{n} box{'es' if n != 1 else ''}")

    def _on_load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images (multiple allowed)", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if not paths:
            return
        self._image_paths = [Path(p) for p in sorted(paths)]
        self._current_idx = 0
        self._low_conf_log.clear()
        self._refresh_log_panel()
        self._goto(0)

    def _on_set_save_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save YOLO Labels")
        if folder:
            self._save_dir = Path(folder)
            display = str(self._save_dir)
            if len(display) > 30:
                display = "..." + display[-27:]
            self.lbl_save_dir.setText(f"Folder: {display}")
            self._set_status(f"Save folder set: {folder}")
            self.btn_save_current.setEnabled(len(self.canvas.get_bboxes()) > 0)

    def _on_save_current(self):
        if self._image_bgr is None:
            self._set_status("No image loaded.")
            return
        if not self._save_dir:
            QMessageBox.warning(self, "Warning", "Set a save directory first!")
            return

        bboxes = self.canvas.get_bboxes()
        to_save = [b for b in bboxes if b.score >= self._threshold]
        skipped = len(bboxes) - len(to_save)
        if not to_save:
            self._set_status("No bboxes above threshold to save.")
            return

        img_path = self._image_paths[self._current_idx]
        label_path = self._save_dir / (img_path.stem + ".txt")
        ih, iw = self._image_bgr.shape[:2]
        lines = [b.to_yolo(iw, ih) for b in to_save]
        label_path.write_text("\n".join(lines))

        msg = f"Saved: {label_path.name}  ({len(lines)} bbox"
        if skipped > 0:
            msg += f", {skipped} low-conf skipped"
        msg += ")"
        self._set_status(msg)
        QMessageBox.information(self, "Saved", msg)

    def _auto_save(self):
        if self._image_bgr is None or not self._save_dir:
            return
        bboxes = self.canvas.get_bboxes()
        to_save = [b for b in bboxes if b.score >= self._threshold]
        if not to_save:
            return
        img_path = self._image_paths[self._current_idx]
        label_path = self._save_dir / (img_path.stem + ".txt")
        ih, iw = self._image_bgr.shape[:2]
        label_path.write_text("\n".join(b.to_yolo(iw, ih) for b in to_save))

    def _on_run_detection(self):
        if self._image_bgr is None:
            self._set_status("Load an image first!")
            return
        if self._detector is None:
            QMessageBox.warning(self, "No Model",
                                "No ONNX model loaded.\n"
                                "Use 'Change ONNX Model' to load a .onnx file.")
            return

        padding = self.spin_padding.value()
        dets = self._detector.detect(self._image_bgr, padding)
        new_bboxes = [BBox(x1, y1, x2, y2, score) for x1, y1, x2, y2, score in dets]
        self.canvas.set_bboxes(new_bboxes)

        if not dets:
            self._set_status("No objects detected.")
            return

        threshold = self._threshold
        best_score = dets[0][4]
        icon = "✓" if best_score >= threshold else "⚠"
        save_note = "will be saved" if best_score >= threshold else "LOW CONF — will NOT be saved"
        self._set_status(
            f"Detected 1 best bbox  |  conf: {best_score:.3f}  {icon}  ({save_note})")

    def _on_clear_bboxes(self):
        self.canvas.clear_bboxes()

    def _on_crop_selected(self):
        if self._image_bgr is None:
            return
        bboxes = self.canvas.get_bboxes()
        sel = [b for b in bboxes if b.selected]
        if not sel:
            self._set_status("Select a bbox first!")
            return
        for b in sel:
            crop = self._image_bgr[b.y1:b.y2, b.x1:b.x2]
            if crop.size == 0:
                continue
            dlg = CropDialog(crop, b, self._threshold, self)
            dlg.exec_()
        self._set_status(f"Crop preview: {len(sel)} bbox")

    def _on_copy_selected(self):
        if self._image_bgr is None:
            return
        bboxes = self.canvas.get_bboxes()
        sel = [b for b in bboxes if b.selected]
        if not sel:
            self._set_status("Select a bbox first!")
            return

        ih, iw = self._image_bgr.shape[:2]
        norm = []
        for b in sel:
            cls_id, cx_n, cy_n, w_n, h_n = b.to_yolo_normalized(iw, ih)
            norm.append((cls_id, cx_n, cy_n, w_n, h_n, b.score))

        src_name = self._image_paths[self._current_idx].name if self._image_paths else "?"
        self._clipboard = {
            "source_idx": self._current_idx,
            "source_name": src_name,
            "bboxes": norm,
        }
        self.lbl_clipboard.setText(
            f"{len(norm)} bbox from\n[{self._current_idx+1}] {src_name}")
        self.lbl_clipboard.setStyleSheet(
            f"color: {SUCCESS}; font-size: 9px; font-family: Consolas;"
            f"background: {CARD}; border: 1px solid #2d2d3d;"
            "border-radius: 4px; padding: 5px 7px;")
        self.btn_paste_current.setEnabled(True)
        self.btn_paste_next.setEnabled(True)
        self._set_status(f"Copied {len(norm)} bbox from [{self._current_idx+1}] {src_name}")

    def _on_paste_to_current(self):
        if not self._clipboard or not self._clipboard.get("bboxes"):
            QMessageBox.warning(self, "Clipboard Empty", "Copy bboxes first!")
            return
        if self._image_bgr is None:
            return

        ih, iw = self._image_bgr.shape[:2]
        norm = self._clipboard["bboxes"]
        new_bboxes = [
            BBox.from_yolo_normalized(cls_id, cx_n, cy_n, w_n, h_n, iw, ih, score=s)
            for (cls_id, cx_n, cy_n, w_n, h_n, s) in norm
        ]
        existing = self.canvas.get_bboxes()
        self.canvas.set_bboxes(existing + new_bboxes)
        self._set_status(f"{len(new_bboxes)} bbox pasted to current image")

    def _on_paste_to_next(self):
        if not self._clipboard or not self._clipboard.get("bboxes"):
            QMessageBox.warning(self, "Clipboard Empty", "Copy bboxes first!")
            return
        if not self._save_dir:
            QMessageBox.warning(self, "No Save Directory", "Set a save directory first!")
            return
        target = self._current_idx + 1
        if target >= len(self._image_paths):
            self._set_status("No next image available!")
            return

        target_path = self._image_paths[target]
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            return
        t_h, t_w = target_img.shape[:2]
        norm = self._clipboard["bboxes"]
        new_bboxes = [
            BBox.from_yolo_normalized(cls_id, cx_n, cy_n, w_n, h_n, t_w, t_h, score=s)
            for (cls_id, cx_n, cy_n, w_n, h_n, s) in norm
        ]
        label_path = self._save_dir / (target_path.stem + ".txt")
        existing_lines = []
        if label_path.exists():
            existing_lines = [l for l in label_path.read_text().strip().splitlines() if l.strip()]
        new_lines = [b.to_yolo(t_w, t_h) for b in new_bboxes]
        label_path.write_text("\n".join(existing_lines + new_lines))
        self._set_status(
            f"{len(new_bboxes)} bbox pasted to [{target+1}] {target_path.name}")

    def _on_threshold_change(self, value: float):
        self._threshold = value
        self.canvas.set_threshold(value)
        self._check_low_confidence()
        self._update_tree()

    def _check_low_confidence(self):
        if self._image_bgr is None or not self._image_paths:
            return
        bboxes = self.canvas.get_bboxes()
        low = [b for b in bboxes if b.score < self._threshold]
        idx = self._current_idx
        name = self._image_paths[idx].name if self._image_paths else "?"

        if low:
            min_score = min(b.score for b in low)
            self._low_conf_log[idx] = {"name": name, "count": len(low), "min_score": min_score}
        else:
            self._low_conf_log.pop(idx, None)

        self._refresh_log_panel()

    def _refresh_log_panel(self):
        self.log_list.clear()
        for img_idx in sorted(self._low_conf_log.keys()):
            entry = self._low_conf_log[img_idx]
            marker = "▶ " if img_idx == self._current_idx else "   "
            text = (f"{marker}[{img_idx+1:03d}] {entry['name']}\n"
                    f"        {entry['count']} bbox  |  min={entry['min_score']:.3f}")
            item = QListWidgetItem(text)
            color = QColor(DANGER) if entry["min_score"] < 0.3 else QColor(WARN)
            item.setForeground(color)
            item.setData(Qt.UserRole, img_idx)
            self.log_list.addItem(item)

        total = len(self._low_conf_log)
        if total:
            self.lbl_log_count.setText(
                f"{total} problematic image{'s' if total > 1 else ''}")
            self.lbl_log_count.setStyleSheet(
                f"color: {WARN}; font-size: 10px; font-weight: bold; font-family: Consolas;")
        else:
            self.lbl_log_count.setText("✓  All confidence OK")
            self.lbl_log_count.setStyleSheet(
                f"color: {SUCCESS}; font-size: 10px; font-weight: bold; font-family: Consolas;")

    def _on_log_click(self, item: QListWidgetItem):
        img_idx = item.data(Qt.UserRole)
        if img_idx is not None:
            self._goto(img_idx, auto_save_current=True)

    def _on_clear_log(self):
        self._low_conf_log.clear()
        self._refresh_log_panel()
        self._set_status("Low confidence log cleared.")

    def _navigate(self, direction: int):
        if not self._image_paths:
            return
        new_idx = self._current_idx + direction
        if 0 <= new_idx < len(self._image_paths):
            self._goto(new_idx, auto_save_current=True)

    def _goto(self, idx: int, auto_save_current: bool = False):
        if not self._image_paths or not (0 <= idx < len(self._image_paths)):
            return

        if auto_save_current and self._current_idx >= 0:
            self._auto_save()

        self._current_idx = idx
        path = self._image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            self._set_status(f"Failed to read: {path.name}")
            return

        self._image_bgr = img
        self.canvas.load(img)
        self.canvas.set_threshold(self._threshold)

        if self._save_dir:
            label_path = self._save_dir / (path.stem + ".txt")
            if label_path.exists():
                self._load_existing_labels(label_path)

        n = len(self._image_paths)
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < n - 1)
        self.lbl_counter.setText(f"{idx+1} / {n}")

        fname_display = path.name
        if len(fname_display) > 26:
            fname_display = fname_display[:12] + "…" + fname_display[-11:]
        self.lbl_filename.setText(fname_display)

        self.btn_detect.setEnabled(True)
        self.btn_clear.setEnabled(False)
        self.btn_crop.setEnabled(False)
        self.btn_copy.setEnabled(False)
        self.btn_paste_current.setEnabled(self._clipboard is not None)
        self.btn_paste_next.setEnabled(self._clipboard is not None)
        self.btn_save_current.setEnabled(self._save_dir is not None)

        h, w = img.shape[:2]
        self._set_status(f"[{idx+1}/{n}] {path.name}  ({w}×{h})")
        self._check_low_confidence()

    def _load_existing_labels(self, label_path: Path):
        if self._image_bgr is None:
            return
        ih, iw = self._image_bgr.shape[:2]
        bboxes = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * iw)
            y1 = int((cy - h / 2) * ih)
            x2 = int((cx + w / 2) * iw)
            y2 = int((cy + h / 2) * ih)
            bboxes.append(BBox(x1, y1, x2, y2, score=1.0, class_id=cls))
        self.canvas.set_bboxes(bboxes)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())