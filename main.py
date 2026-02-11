#!/usr/bin/env python3
"""
AGC LOGO MARK - Real-Time OCR Application
Optimized for NVIDIA Jetson Orin Nano with CUDA acceleration.
Uses EasyOCR (PyTorch + CUDA) for real-time text detection.
"""

import sys
import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import easyocr

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QLineEdit, QScrollArea,
    QFrame, QSplitter, QDialog, QDialogButtonBox, QFormLayout,
    QMessageBox, QComboBox, QSpinBox, QTextEdit, QGroupBox,
    QSizePolicy, QHeaderView, QStatusBar, QProgressBar,
    QSlider, QCheckBox, QMenu
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot, QSize, QMutex, QMutexLocker
)
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QIcon, QPalette, QPainter,
    QBrush, QPen, QAction
)

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"ocr_{datetime.now():%Y%m%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("AGC_OCR")

# ──────────────────────────────────────────────────────────────
# Dark theme stylesheet
# ──────────────────────────────────────────────────────────────
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Noto Sans', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 6px;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 10px 28px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
    font-size: 14px;
    min-width: 120px;
}
QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QTabBar::tab:hover:!selected {
    background-color: #45475a;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 13px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #b4d0fb;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#dangerBtn {
    background-color: #f38ba8;
}
QPushButton#dangerBtn:hover {
    background-color: #f5a3b8;
}
QPushButton#successBtn {
    background-color: #a6e3a1;
}
QPushButton#successBtn:hover {
    background-color: #b8eab4;
}
QPushButton#warningBtn {
    background-color: #f9e2af;
    color: #1e1e2e;
}
QPushButton#warningBtn:hover {
    background-color: #fbe8c0;
}
QPushButton#jobBtn {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
    padding: 18px 10px;
    font-size: 15px;
    min-height: 50px;
}
QPushButton#jobBtn:hover {
    border-color: #89b4fa;
    background-color: #45475a;
}
QPushButton#jobBtnActive {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: 2px solid #89b4fa;
    padding: 18px 10px;
    font-size: 15px;
    min-height: 50px;
    font-weight: bold;
}
QTableWidget {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 6px;
    gridline-color: #313244;
    selection-background-color: #45475a;
}
QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #313244;
}
QHeaderView::section {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px;
    border: none;
    font-weight: bold;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    color: #cdd6f4;
    font-size: 13px;
}
QLineEdit:focus, QTextEdit:focus {
    border-color: #89b4fa;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    font-size: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}
QScrollArea {
    border: none;
}
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    font-size: 12px;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    background-color: #313244;
    color: #cdd6f4;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QLabel#titleLabel {
    font-size: 20px;
    font-weight: bold;
    color: #89b4fa;
}
QLabel#statusOn {
    color: #a6e3a1;
    font-weight: bold;
}
QLabel#statusOff {
    color: #f38ba8;
    font-weight: bold;
}
QLabel#headerLabel {
    font-size: 16px;
    font-weight: bold;
    color: #cdd6f4;
    padding: 4px 0;
}
QFrame#separator {
    background-color: #45475a;
    max-height: 1px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #45475a;
    border-radius: 4px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
"""


# ──────────────────────────────────────────────────────────────
# Relay Worker Thread
# ──────────────────────────────────────────────────────────────
class RelayWorker(QThread):
    """Monitors relay board DI signals in a background thread."""
    di1_triggered = Signal()   # (unused)
    di2_triggered = Signal()   # OCR trigger
    relay_connected = Signal(bool)
    relay_error = Signal(str)

    def __init__(self, host="192.168.1.200", port=502):
        super().__init__()
        self.host = host
        self.port = port
        self._running = False
        self._relay = None
        self._prev_di1 = False
        self._prev_di2 = False

    def run(self):
        self._running = True
        try:
            from Relay_B import Relay
            self._relay = Relay(host=self.host, port=self.port)
            self._relay.connect()
            self.relay_connected.emit(True)
            logger.info(f"Relay connected at {self.host}:{self.port}")
        except Exception as e:
            self.relay_error.emit(str(e))
            self.relay_connected.emit(False)
            logger.warning(f"Relay connection failed: {e}")
            self._running = False
            return

        while self._running:
            try:
                di1 = self._relay.is_DI_on(1)
                di2 = self._relay.is_DI_on(2)

                # Rising edge detection
                if di2 and not self._prev_di2:
                    self.di2_triggered.emit()
                    logger.info("DI2 triggered - OCR start")

                self._prev_di1 = di1
                self._prev_di2 = di2
            except Exception as e:
                self.relay_error.emit(str(e))
                logger.error(f"Relay read error: {e}")

            self.msleep(100)

        if self._relay:
            try:
                self._relay.disconnect()
            except Exception:
                pass

    def stop(self):
        self._running = False
        self.wait(2000)

    def turn_on_relay(self, channel: int):
        if self._relay:
            try:
                self._relay.on(channel)
            except Exception as e:
                self.relay_error.emit(str(e))

    def turn_off_relay(self, channel: int):
        if self._relay:
            try:
                self._relay.off(channel)
            except Exception as e:
                self.relay_error.emit(str(e))


# ──────────────────────────────────────────────────────────────
# OCR Worker Thread
# ──────────────────────────────────────────────────────────────
class OCRWorker(QThread):
    """Performs OCR processing in a background thread to avoid UI blocking."""
    ocr_result = Signal(list, object)  # (results, annotated_frame)
    ocr_error = Signal(str)
    ocr_busy = Signal(bool)

    def __init__(self, languages=None, use_gpu=True):
        super().__init__()
        self._frame = None
        self._running = False
        self._mutex = QMutex()
        self._process_requested = False
        self._reader = None
        self._languages = languages or ["en"]
        self._use_gpu = use_gpu
        self._rotation_angle = 0

    def init_reader(self):
        """Initialize EasyOCR reader (call from thread)."""
        try:
            logger.info(f"Initializing EasyOCR (GPU={self._use_gpu}, langs={self._languages})...")
            self._reader = easyocr.Reader(
                self._languages,
                gpu=self._use_gpu,
                model_storage_directory=str(Path(__file__).parent / "ocr_models"),
            )
            logger.info("EasyOCR initialized successfully")
            # Log GPU info
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"CUDA GPU: {gpu_name}")
                else:
                    logger.warning("CUDA not available, using CPU")
            except ImportError:
                pass
        except Exception as e:
            self.ocr_error.emit(f"Failed to initialize OCR: {e}")
            logger.error(f"OCR init error: {e}")

    def set_rotation(self, angle: int):
        self._rotation_angle = angle

    def request_process(self, frame: np.ndarray):
        """Queue a frame for OCR processing."""
        with QMutexLocker(self._mutex):
            self._frame = frame.copy()
            self._process_requested = True

    def run(self):
        self._running = True
        self.init_reader()

        while self._running:
            process = False
            frame = None

            with QMutexLocker(self._mutex):
                if self._process_requested and self._frame is not None:
                    frame = self._frame.copy()
                    self._process_requested = False
                    process = True

            if process and self._reader is not None and frame is not None:
                self.ocr_busy.emit(True)
                try:
                    processed = frame.copy()

                    # Apply rotation if needed
                    if self._rotation_angle != 0:
                        h, w = processed.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, self._rotation_angle, 1.0)
                        processed = cv2.warpAffine(processed, M, (w, h))

                    # Run OCR
                    results = self._reader.readtext(processed)

                    # Draw bounding boxes on frame
                    annotated = processed.copy()
                    for detection in results:
                        det = list(detection)
                        bbox = det[0]
                        det_text = str(det[1])
                        det_conf = float(det[2])
                        pts = np.array(bbox, dtype=np.int32)

                        # Draw rotated rectangle with colored border
                        rect = cv2.minAreaRect(pts)
                        box_pts = np.array(cv2.boxPoints(rect), dtype=np.int32)

                        # Color based on confidence
                        if det_conf > 0.8:
                            color = (0, 255, 0)      # green
                        elif det_conf > 0.5:
                            color = (0, 200, 255)    # yellow/orange
                        else:
                            color = (0, 0, 255)      # red

                        # Draw filled semi-transparent background for the box
                        overlay = annotated.copy()
                        cv2.fillPoly(overlay, [box_pts], color)
                        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

                        # Draw rotated rectangle border
                        cv2.drawContours(annotated, [box_pts], 0, color, 2)

                        # Draw corner dots
                        for i in range(len(box_pts)):
                            pt = (int(box_pts[i][0]), int(box_pts[i][1]))
                            cv2.circle(annotated, pt, 4, color, -1)

                        # Label with background
                        label = f"{det_text} ({det_conf:.0%})"
                        x, y = int(pts[0][0]), int(pts[0][1]) - 12
                        y = max(y, 25)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.55
                        thickness = 1
                        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        # Background rectangle for text
                        cv2.rectangle(
                            annotated,
                            (x - 2, y - th - 6),
                            (x + tw + 4, y + 4),
                            color, -1
                        )
                        cv2.putText(
                            annotated, label, (x, y - 2),
                            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
                        )

                    self.ocr_result.emit(results, annotated)
                    logger.info(f"OCR detected {len(results)} text regions")
                except Exception as e:
                    self.ocr_error.emit(str(e))
                    logger.error(f"OCR processing error: {e}")
                finally:
                    self.ocr_busy.emit(False)

            self.msleep(50)

    def stop(self):
        self._running = False
        self.wait(5000)


# ──────────────────────────────────────────────────────────────
# Add New Job Dialog
# ──────────────────────────────────────────────────────────────
class AddJobDialog(QDialog):
    """Dialog for creating a new job (JSON file)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Job")
        self.setMinimumWidth(450)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("Create New Inspection Job")
        title.setObjectName("headerLabel")
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(10)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. ProductA_Logo")
        form.addRow("Job Name:", self.name_input)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Enter text patterns to detect, one per line.\n"
            "Example:\nTOYOTA\nMADE IN JAPAN\n2025"
        )
        self.text_input.setMaximumHeight(180)
        form.addRow("Detection Texts:", self.text_input)

        layout.addLayout(form)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_data(self):
        name = self.name_input.text().strip()
        texts = [
            line.strip()
            for line in self.text_input.toPlainText().strip().split("\n")
            if line.strip()
        ]
        return name, texts


# ──────────────────────────────────────────────────────────────
# Main Application Window
# ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGC LOGO MARK - Real-Time OCR Inspector")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "logo.ico")))

        # State
        self._app_dir = Path(__file__).parent
        self._current_job_file: Optional[str] = None
        self._model_data: list = []
        self._ocr_results: list = []
        self._camera_index = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_inspecting = False
        self._relay_connected = False
        self._rotation_angle = 0
        self._auto_trigger = False
        self._continuous_ocr = False
        self._frame_paused = False
        self._frozen_frame = None

        # Log buffer for Log tab
        self._log_handler = LogHandler()
        logging.getLogger("AGC_OCR").addHandler(self._log_handler)

        self._build_ui()
        self._setup_workers()
        self._setup_camera()
        self._connect_signals()

        # Camera timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(30)

        # Log refresh timer
        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._refresh_log)
        self._log_timer.start(1000)

        logger.info("Application started")
        self.statusBar().showMessage("Ready  |  Camera initializing...")

    # ── UI Construction ──────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 4)
        main_layout.setSpacing(6)

        # Top bar
        top_bar = QHBoxLayout()
        title = QLabel("AGC LOGO MARK")
        title.setObjectName("titleLabel")
        top_bar.addWidget(title)
        top_bar.addStretch()

        # GPU indicator
        self.gpu_label = QLabel()
        self._update_gpu_label()
        top_bar.addWidget(self.gpu_label)

        # Relay indicator
        self.relay_indicator = QLabel("  RELAY: Disconnected  ")
        self.relay_indicator.setObjectName("statusOff")
        top_bar.addWidget(self.relay_indicator)

        main_layout.addLayout(top_bar)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        main_layout.addWidget(sep)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)

        self._build_monitor_tab()
        self._build_job_change_tab()
        self._build_setting_tab()
        self._build_log_tab()

        # Status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _update_gpu_label(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.gpu_label.setText(f"  GPU: {name}  ")
                self.gpu_label.setObjectName("statusOn")
            else:
                self.gpu_label.setText("  GPU: CPU Mode  ")
                self.gpu_label.setObjectName("statusOff")
        except ImportError:
            self.gpu_label.setText("  GPU: N/A  ")
            self.gpu_label.setObjectName("statusOff")

    # ── Monitor Tab ──────────────────────────────────────────
    def _build_monitor_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "  Monitor  ")
        layout = QHBoxLayout(tab)
        layout.setSpacing(12)

        # Left: Video + Controls
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #11111b; border: 2px solid #45475a; border-radius: 8px;"
        )
        self.video_label.setText("Camera Feed")
        left_panel.addWidget(self.video_label, stretch=1)

        # Controls row
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)

        self.btn_trigger = QPushButton("  Trigger OCR")
        self.btn_trigger.setObjectName("successBtn")
        self.btn_trigger.clicked.connect(self._trigger_ocr)
        ctrl_layout.addWidget(self.btn_trigger)

        self.btn_stop = QPushButton("  Stop")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.clicked.connect(self._stop_inspection)
        self.btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self.btn_stop)

        self.btn_continuous = QCheckBox("Continuous OCR")
        self.btn_continuous.toggled.connect(self._toggle_continuous)
        ctrl_layout.addWidget(self.btn_continuous)

        ctrl_layout.addStretch()

        # Rotation control
        rot_label = QLabel("Rotation:")
        ctrl_layout.addWidget(rot_label)
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setMaximumWidth(150)
        self.rotation_slider.valueChanged.connect(self._on_rotation_changed)
        ctrl_layout.addWidget(self.rotation_slider)
        self.rotation_value = QLabel("0°")
        self.rotation_value.setMinimumWidth(40)
        ctrl_layout.addWidget(self.rotation_value)

        btn_reset_rot = QPushButton("Reset")
        btn_reset_rot.setMaximumWidth(100)
        btn_reset_rot.clicked.connect(lambda: self.rotation_slider.setValue(0))
        ctrl_layout.addWidget(btn_reset_rot)

        left_panel.addLayout(ctrl_layout)
        layout.addLayout(left_panel, stretch=3)

        # Right: Results panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # Current job info
        job_group = QGroupBox("Current Job")
        job_layout = QVBoxLayout(job_group)
        self.current_job_label = QLabel("No job loaded")
        self.current_job_label.setObjectName("headerLabel")
        self.current_job_label.setWordWrap(True)
        job_layout.addWidget(self.current_job_label)

        btn_load_row = QHBoxLayout()
        self.btn_load_model = QPushButton("Load Job File")
        self.btn_load_model.clicked.connect(self._load_model_file)
        btn_load_row.addWidget(self.btn_load_model)
        job_layout.addLayout(btn_load_row)
        right_panel.addWidget(job_group)

        # Results table
        results_group = QGroupBox("OCR Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(
            ["Expected Text", "Detected", "Status"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(False)
        results_layout.addWidget(self.results_table)

        # Edit/Save buttons
        edit_row = QHBoxLayout()
        self.btn_edit_results = QPushButton("Edit")
        self.btn_edit_results.setObjectName("warningBtn")
        self.btn_edit_results.clicked.connect(self._edit_model_data)
        self.btn_edit_results.setEnabled(False)
        edit_row.addWidget(self.btn_edit_results)

        self.btn_save_results = QPushButton("Save Changes")
        self.btn_save_results.setObjectName("successBtn")
        self.btn_save_results.clicked.connect(self._save_model_data)
        self.btn_save_results.setEnabled(False)
        edit_row.addWidget(self.btn_save_results)
        results_layout.addLayout(edit_row)

        right_panel.addWidget(results_group, stretch=1)

        # Detection summary
        summary_group = QGroupBox("Detection Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("No results yet")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignTop)
        summary_layout.addWidget(self.summary_label)
        right_panel.addWidget(summary_group)

        layout.addLayout(right_panel, stretch=2)

    # ── Job Change Tab ───────────────────────────────────────
    def _build_job_change_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "  Job Change  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        lbl = QLabel("Select Inspection Job")
        lbl.setObjectName("headerLabel")
        header.addWidget(lbl)
        header.addStretch()

        btn_refresh = QPushButton("Refresh Jobs")
        btn_refresh.clicked.connect(self._refresh_jobs)
        header.addWidget(btn_refresh)

        btn_add = QPushButton("Add New Job")
        btn_add.setObjectName("successBtn")
        btn_add.clicked.connect(self._add_new_job)
        header.addWidget(btn_add)

        layout.addLayout(header)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Scrollable job buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.job_container = QWidget()
        self.job_grid = QGridLayout(self.job_container)
        self.job_grid.setSpacing(12)
        self.job_grid.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(self.job_container)
        layout.addWidget(scroll, stretch=1)

        self._refresh_jobs()

    # ── Setting Tab ──────────────────────────────────────────
    def _build_setting_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "  Settings  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # ── Camera Settings ──
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QFormLayout(cam_group)
        cam_layout.setSpacing(10)

        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 10)
        self.cam_index_spin.setValue(0)
        cam_layout.addRow("Camera Index:", self.cam_index_spin)

        btn_apply_cam = QPushButton("Apply Camera")
        btn_apply_cam.clicked.connect(self._apply_camera_settings)
        cam_layout.addRow("", btn_apply_cam)
        layout.addWidget(cam_group)

        # ── Relay Settings ──
        relay_group = QGroupBox("Relay Board Settings")
        relay_layout = QFormLayout(relay_group)
        relay_layout.setSpacing(10)

        self.relay_host_input = QLineEdit("192.168.1.200")
        relay_layout.addRow("Relay IP:", self.relay_host_input)

        self.relay_port_spin = QSpinBox()
        self.relay_port_spin.setRange(1, 65535)
        self.relay_port_spin.setValue(502)
        relay_layout.addRow("Relay Port:", self.relay_port_spin)

        relay_btn_row = QHBoxLayout()
        self.btn_connect_relay = QPushButton("Connect Relay")
        self.btn_connect_relay.clicked.connect(self._connect_relay)
        relay_btn_row.addWidget(self.btn_connect_relay)

        self.btn_disconnect_relay = QPushButton("Disconnect")
        self.btn_disconnect_relay.setObjectName("dangerBtn")
        self.btn_disconnect_relay.clicked.connect(self._disconnect_relay)
        self.btn_disconnect_relay.setEnabled(False)
        relay_btn_row.addWidget(self.btn_disconnect_relay)
        relay_layout.addRow("", relay_btn_row)
        layout.addWidget(relay_group)

        # ── OCR Settings ──
        ocr_group = QGroupBox("OCR Engine Settings")
        ocr_layout = QFormLayout(ocr_group)
        ocr_layout.setSpacing(10)

        self.gpu_checkbox = QCheckBox("Use GPU (CUDA)")
        self.gpu_checkbox.setChecked(True)
        ocr_layout.addRow("Acceleration:", self.gpu_checkbox)

        info_label = QLabel(
            "EasyOCR uses PyTorch with CUDA for GPU acceleration.\n"
            "On Jetson Orin Nano, ensure JetPack SDK is installed.\n"
            "GPU mode significantly speeds up text detection."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #a6adc8; font-style: italic;")
        ocr_layout.addRow("", info_label)
        layout.addWidget(ocr_group)

        layout.addStretch()

    # ── Log Tab ──────────────────────────────────────────────
    def _build_log_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "  Log  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        header = QHBoxLayout()
        lbl = QLabel("Application Log")
        lbl.setObjectName("headerLabel")
        header.addWidget(lbl)
        header.addStretch()

        btn_clear = QPushButton("Clear Log")
        btn_clear.setObjectName("dangerBtn")
        btn_clear.clicked.connect(self._clear_log)
        header.addWidget(btn_clear)

        btn_export = QPushButton("Export Log")
        btn_export.clicked.connect(self._export_log)
        header.addWidget(btn_export)

        layout.addLayout(header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;"
        )
        layout.addWidget(self.log_text)

    # ── Workers setup ────────────────────────────────────────
    def _setup_workers(self):
        # OCR worker
        self.ocr_worker = OCRWorker(use_gpu=True)
        self.ocr_worker.start()

        # Relay worker (not started until user connects)
        self.relay_worker = None

    def _setup_camera(self):
        """Initialize camera capture."""
        if self._capture is not None:
            self._capture.release()

        self._capture = cv2.VideoCapture(self._camera_index)
        if self._capture.isOpened():
            # Try setting resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera {self._camera_index} opened: {w}x{h}")
            self.statusBar().showMessage(f"Camera {self._camera_index} active ({w}x{h})")
        else:
            logger.warning(f"Failed to open camera {self._camera_index}")
            self.statusBar().showMessage("Camera not available")

    def _connect_signals(self):
        self.ocr_worker.ocr_result.connect(self._on_ocr_result)
        self.ocr_worker.ocr_error.connect(self._on_ocr_error)
        self.ocr_worker.ocr_busy.connect(self._on_ocr_busy)

    # ── Frame Update ─────────────────────────────────────────
    @Slot()
    def _update_frame(self):
        if self._capture is None or not self._capture.isOpened():
            return

        # If frame is paused (waiting for OCR), show frozen frame
        if self._frame_paused and self._frozen_frame is not None:
            return

        ret, frame = self._capture.read()
        if not ret:
            return

        # Apply rotation for display
        display_frame = frame.copy()
        if self._rotation_angle != 0:
            h, w = display_frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self._rotation_angle, 1.0)
            display_frame = cv2.warpAffine(display_frame, M, (w, h))

        self._display_frame(display_frame)

        # Continuous OCR mode
        if self._continuous_ocr and self._is_inspecting:
            self.ocr_worker.request_process(frame)

    def _display_frame(self, frame: np.ndarray):
        """Convert OpenCV frame to QPixmap and display."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        # Scale to fit label while maintaining aspect ratio
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    # ── OCR Trigger ──────────────────────────────────────────
    @Slot()
    def _trigger_ocr(self):
        if self._capture is None or not self._capture.isOpened():
            self.statusBar().showMessage("No camera available")
            return

        ret, frame = self._capture.read()
        if not ret:
            self.statusBar().showMessage("Failed to capture frame")
            return

        self._is_inspecting = True
        self._frame_paused = True
        self.btn_trigger.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Freeze and display the captured frame
        frozen_display = frame.copy()
        if self._rotation_angle != 0:
            h, w = frozen_display.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self._rotation_angle, 1.0)
            frozen_display = cv2.warpAffine(frozen_display, M, (w, h))
        self._frozen_frame = frozen_display
        # Draw "Processing..." overlay on frozen frame
        overlay = frozen_display.copy()
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frozen_display, 0.4, 0, overlay)
        cv2.putText(overlay, "OCR Processing...", (w // 2 - 150, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        self._display_frame(overlay)

        self.ocr_worker.set_rotation(self._rotation_angle)
        self.ocr_worker.request_process(frame)
        self.statusBar().showMessage("Processing OCR...")
        logger.info("OCR triggered manually")

    @Slot()
    def _stop_inspection(self):
        self._is_inspecting = False
        self._continuous_ocr = False
        self._frame_paused = False
        self._frozen_frame = None
        self.btn_continuous.setChecked(False)
        self.btn_trigger.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Turn off relay channels on stop
        if self._relay_connected and self.relay_worker:
            self.relay_worker.turn_off_relay(4)
            self.relay_worker.turn_off_relay(5)
        self.statusBar().showMessage("Inspection stopped")
        logger.info("Inspection stopped")

    def _toggle_continuous(self, checked: bool):
        self._continuous_ocr = checked
        if checked:
            self._is_inspecting = True
            self.btn_stop.setEnabled(True)
            self.statusBar().showMessage("Continuous OCR mode active")
            logger.info("Continuous OCR enabled")
        else:
            self._is_inspecting = False
            self.btn_trigger.setEnabled(True)
            self.btn_stop.setEnabled(False)

    # ── OCR Result Handling ──────────────────────────────────
    @Slot(list, object)
    def _on_ocr_result(self, results: list, annotated_frame):
        """Handle OCR results from worker thread."""
        self._ocr_results = results

        # Display annotated frame (keep paused to show results)
        if annotated_frame is not None:
            self._frozen_frame = annotated_frame
            self._display_frame(annotated_frame)

        # Un-pause frame only if not in continuous mode
        if not self._continuous_ocr:
            self._frame_paused = True  # stay paused to show result overlay

        # Compare with model data
        detected_texts = [text.upper().strip() for (_, text, _) in results]

        self.results_table.setRowCount(0)

        if self._model_data:
            self.results_table.setRowCount(len(self._model_data))
            match_count = 0
            partial_count = 0

            for row, expected in enumerate(self._model_data):
                expected_upper = expected.upper().strip()

                # Expected text (editable)
                item_expected = QTableWidgetItem(expected)
                item_expected.setFlags(
                    item_expected.flags() | Qt.ItemIsEditable
                )
                self.results_table.setItem(row, 0, item_expected)

                # Find match across ALL detected texts
                best_match = ""
                status = "NOT FOUND"
                status_color = QColor("#f38ba8")  # red

                for det in detected_texts:
                    if expected_upper == det:
                        # Exact match
                        best_match = det
                        status = "MATCH"
                        status_color = QColor("#a6e3a1")  # green
                        match_count += 1
                        break
                    elif expected_upper in det or det in expected_upper:
                        # Substring match (partial) - treat as MATCH
                        best_match = det
                        status = "MATCH"
                        status_color = QColor("#a6e3a1")  # green
                        match_count += 1
                        break

                # If no single text matched, check by combining all detected texts
                if status == "NOT FOUND" and detected_texts:
                    combined = " ".join(detected_texts)
                    if expected_upper in combined:
                        best_match = combined
                        status = "MATCH"
                        status_color = QColor("#a6e3a1")  # green
                        match_count += 1

                item_detected = QTableWidgetItem(best_match)
                item_detected.setFlags(
                    item_detected.flags() & ~Qt.ItemIsEditable
                )
                self.results_table.setItem(row, 1, item_detected)

                item_status = QTableWidgetItem(status)
                item_status.setFlags(
                    item_status.flags() & ~Qt.ItemIsEditable
                )
                item_status.setForeground(QBrush(status_color))
                item_status.setTextAlignment(Qt.AlignCenter)
                font = item_status.font()
                font.setBold(True)
                item_status.setFont(font)
                self.results_table.setItem(row, 2, item_status)

                # Set text color for expected and detected columns
                item_expected.setForeground(QBrush(status_color))
                item_detected.setForeground(QBrush(status_color))

            total = len(self._model_data)
            not_found = total - match_count - partial_count
            self.summary_label.setText(
                f"Total: {total}  |  "
                f"<span style='color:#a6e3a1;'>Match: {match_count}</span>  |  "
                f"<span style='color:#f9e2af;'>Partial: {partial_count}</span>  |  "
                f"<span style='color:#f38ba8;'>Not Found: {not_found}</span>\n\n"
                f"Detected texts: {', '.join(detected_texts) if detected_texts else 'None'}"
            )
            self.btn_save_results.setEnabled(True)
            self.btn_edit_results.setEnabled(True)

            # Pass/Fail relay control
            if self._relay_connected and self.relay_worker:
                if not_found == 0:
                    # All matched - Relay ch4 ON, ch5 OFF
                    self.relay_worker.turn_on_relay(4)
                    self.relay_worker.turn_off_relay(5)
                    logger.info("All matched - Relay CH4 ON, CH5 OFF")
                    # Auto-stop after 5 seconds
                    QTimer.singleShot(5000, self._stop_inspection)
                else:
                    # Not all matched - Relay ch5 ON, ch4 OFF
                    self.relay_worker.turn_on_relay(5)
                    self.relay_worker.turn_off_relay(4)
                    logger.info("Not all matched - Relay CH5 ON, CH4 OFF")
        else:
            # No model loaded - just show all detected texts
            self.results_table.setRowCount(len(results))
            for row, (bbox, text, conf) in enumerate(results):
                self.results_table.setItem(row, 0, QTableWidgetItem("-"))
                self.results_table.setItem(row, 1, QTableWidgetItem(text))

                conf_item = QTableWidgetItem(f"{conf:.0%}")
                conf_item.setTextAlignment(Qt.AlignCenter)
                if conf > 0.8:
                    conf_item.setBackground(QBrush(QColor("#a6e3a1")))
                elif conf > 0.5:
                    conf_item.setBackground(QBrush(QColor("#f9e2af")))
                else:
                    conf_item.setBackground(QBrush(QColor("#f38ba8")))
                conf_item.setForeground(QBrush(QColor("#1e1e2e")))
                self.results_table.setItem(row, 2, conf_item)

            self.summary_label.setText(
                f"Detected {len(results)} text regions (no job loaded for comparison)"
            )

        if not self._continuous_ocr:
            self.btn_trigger.setEnabled(True)

        self.statusBar().showMessage(
            f"OCR complete: {len(results)} texts detected"
        )

    @Slot(str)
    def _on_ocr_error(self, error: str):
        self.statusBar().showMessage(f"OCR Error: {error}")
        logger.error(f"OCR Error: {error}")
        self.btn_trigger.setEnabled(True)

    @Slot(bool)
    def _on_ocr_busy(self, busy: bool):
        self.progress_bar.setVisible(busy)

    # ── Rotation ─────────────────────────────────────────────
    @Slot(int)
    def _on_rotation_changed(self, value: int):
        self._rotation_angle = value
        self.rotation_value.setText(f"{value}°")
        self.ocr_worker.set_rotation(value)

    # ── Model / Job Loading ──────────────────────────────────
    def clean_raw_text(self, raw_text):
        """Clean raw text by removing prefix before and including first '$',
        and removing suffix from second '$' onwards.
        Example: FOD11850100163$1SRG14R(BRK)-MM-4FIMXA-A7$15 -> SRG14R(BRK)-MM-4FIMXA-A7
        """
        raw_text = raw_text.strip()

        # Find first '$'
        first_dollar = raw_text.find('$')
        if first_dollar == -1:
            return raw_text  # No '$' found, return as is

        # Remove everything up to and including first '$'
        text_after_first = raw_text[first_dollar + 1:]

        # Find second '$'
        second_dollar = text_after_first.find('$')
        if second_dollar == -1:
            return text_after_first  # No second '$', return everything after first

        # Return text between first and second '$'
        return text_after_first[:second_dollar]

    def _load_model_file(self):
        """Show a text input dialog to search for a .json job by name."""
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self, "Load Job File",
            "Enter job name (without .json):",
        )
        if not ok or not text.strip():
            return

        search_name = self.clean_raw_text(text.strip())
        # Search for matching .json file (case-insensitive)
        found_path = None
        for jf in self._app_dir.glob("*.json"):
            if jf.stem.lower() == search_name.lower():
                found_path = str(jf)
                break

        if found_path:
            self._load_job(found_path)
        else:
            # Show "Model not found" dialog with option to add new job
            msg = QMessageBox(self)
            msg.setWindowTitle("Model Not Found")
            msg.setText(f"No job file matching '{search_name}.json' was found.")
            msg.setInformativeText("Would you like to create a new job?")
            msg.setIcon(QMessageBox.Warning)
            btn_add = msg.addButton("Add New Job", QMessageBox.AcceptRole)
            msg.addButton(QMessageBox.Cancel)
            msg.setStyleSheet(DARK_STYLESHEET)
            msg.exec()
            if msg.clickedButton() == btn_add:
                self._add_new_job()

    def _confirm_load_job(self, filepath: str, name: str):
        """Show YES/NO confirmation before loading a job."""
        reply = QMessageBox.question(
            self, "Confirm Job Change",
            f"Load job '{name}' as the current inspection job?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self._load_job(filepath)

    def _load_job(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._model_data = data
            else:
                self._model_data = [str(data)]

            self._current_job_file = filepath
            name = Path(filepath).stem
            self.current_job_label.setText(f"Job: {name}")
            self.statusBar().showMessage(f"Loaded job: {name}")
            logger.info(f"Job loaded: {filepath}")

            # Update results table to show expected texts
            self.results_table.setRowCount(len(self._model_data))
            for row, text in enumerate(self._model_data):
                item = QTableWidgetItem(text)
                self.results_table.setItem(row, 0, item)
                self.results_table.setItem(row, 1, QTableWidgetItem(""))
                self.results_table.setItem(row, 2, QTableWidgetItem("PENDING"))

            self.btn_save_results.setEnabled(True)
            self.btn_edit_results.setEnabled(True)

            # Switch to Monitor tab
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load job:\n{e}")
            logger.error(f"Job load error: {e}")

    def _edit_model_data(self):
        """Open a dialog to edit the current job's expected texts."""
        if not self._current_job_file:
            QMessageBox.warning(self, "Warning", "No job file loaded to edit.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Model - {Path(self._current_job_file).stem}")
        dialog.setMinimumWidth(450)
        dialog.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        title = QLabel(f"Edit: {Path(self._current_job_file).stem}")
        title.setObjectName("headerLabel")
        layout.addWidget(title)

        info = QLabel("Edit detection texts below (one per line):")
        info.setStyleSheet("color: #a6adc8;")
        layout.addWidget(info)

        text_edit = QTextEdit()
        text_edit.setPlainText("\n".join(self._model_data))
        text_edit.setMinimumHeight(200)
        layout.addWidget(text_edit)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.Accepted:
            texts = [
                line.strip()
                for line in text_edit.toPlainText().strip().split("\n")
                if line.strip()
            ]
            if not texts:
                QMessageBox.warning(self, "Warning", "Cannot save empty model.")
                return
            try:
                with open(self._current_job_file, "w", encoding="utf-8") as f:
                    json.dump(texts, f, indent=4, ensure_ascii=False)
                self._model_data = texts
                # Refresh results table
                self.results_table.setRowCount(len(texts))
                for row, t in enumerate(texts):
                    self.results_table.setItem(row, 0, QTableWidgetItem(t))
                    self.results_table.setItem(row, 1, QTableWidgetItem(""))
                    self.results_table.setItem(row, 2, QTableWidgetItem("PENDING"))
                self.statusBar().showMessage("Model updated successfully")
                logger.info(f"Model edited: {self._current_job_file}")
                self._refresh_jobs()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
                logger.error(f"Edit save error: {e}")

    def _save_model_data(self):
        if not self._current_job_file:
            QMessageBox.warning(self, "Warning", "No job file loaded to save.")
            return

        # Collect edited texts from table
        texts = []
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item and item.text().strip() and item.text().strip() != "-":
                texts.append(item.text().strip())

        try:
            with open(self._current_job_file, "w", encoding="utf-8") as f:
                json.dump(texts, f, indent=4, ensure_ascii=False)
            self._model_data = texts
            self.statusBar().showMessage("Job data saved successfully")
            logger.info(f"Job saved: {self._current_job_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
            logger.error(f"Save error: {e}")

    # ── Job Change Tab Actions ───────────────────────────────
    def _refresh_jobs(self):
        # Clear existing buttons
        while self.job_grid.count():
            item = self.job_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Scan for JSON files
        json_files = sorted(self._app_dir.glob("*.json"))

        if not json_files:
            lbl = QLabel("No job files found. Click 'Add New Job' to create one.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #a6adc8; font-style: italic; padding: 40px;")
            self.job_grid.addWidget(lbl, 0, 0, 1, 2)
            return

        for i, jf in enumerate(json_files):
            name = jf.stem
            btn = QPushButton(name)

            if str(jf) == self._current_job_file:
                btn.setObjectName("jobBtnActive")
            else:
                btn.setObjectName("jobBtn")

            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked=False, p=str(jf), n=name: self._confirm_load_job(p, n))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda pos, b=btn, p=str(jf), n=name: self._show_job_context_menu(b, pos, p, n)
            )

            row = i // 2
            col = i % 2
            self.job_grid.addWidget(btn, row, col)

        # Add spacer at bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        last_row = (len(json_files) // 2) + 1
        self.job_grid.addWidget(spacer, last_row, 0, 1, 2)

    def _show_job_context_menu(self, button, pos, filepath: str, name: str):
        """Show right-click context menu on job button."""
        menu = QMenu(self)
        menu.setStyleSheet(DARK_STYLESHEET)
        delete_action = menu.addAction(f"Delete '{name}.json'")
        action = menu.exec(button.mapToGlobal(pos))
        if action == delete_action:
            self._delete_job(filepath, name)

    def _delete_job(self, filepath: str, name: str):
        """Delete a job .json file after YES/Cancel confirmation."""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete '{name}.json'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            try:
                os.remove(filepath)
                logger.info(f"Deleted job file: {filepath}")
                self.statusBar().showMessage(f"Job '{name}' deleted")
                # Clear current job if it was the deleted one
                if self._current_job_file == filepath:
                    self._current_job_file = None
                    self._model_data = []
                    self.current_job_label.setText("No job loaded")
                    self.results_table.setRowCount(0)
                self._refresh_jobs()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete:\n{e}")
                logger.error(f"Delete error: {e}")

    def _add_new_job(self):
        dialog = AddJobDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name, texts = dialog.get_data()
            if not name:
                QMessageBox.warning(self, "Warning", "Please enter a job name.")
                return
            if not texts:
                QMessageBox.warning(
                    self, "Warning", "Please enter at least one text pattern."
                )
                return

            filepath = self._app_dir / f"{name}.json"
            if filepath.exists():
                reply = QMessageBox.question(
                    self, "Confirm",
                    f"'{name}.json' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(texts, f, indent=4, ensure_ascii=False)
                logger.info(f"New job created: {filepath}")
                self._refresh_jobs()
                self.statusBar().showMessage(f"Job '{name}' created")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create job:\n{e}")

    # ── Settings Actions ─────────────────────────────────────
    def _apply_camera_settings(self):
        self._camera_index = self.cam_index_spin.value()
        self._setup_camera()

    def _connect_relay(self):
        host = self.relay_host_input.text().strip()
        port = self.relay_port_spin.value()

        if self.relay_worker is not None:
            self.relay_worker.stop()

        self.relay_worker = RelayWorker(host=host, port=port)
        self.relay_worker.relay_connected.connect(self._on_relay_connected)
        self.relay_worker.relay_error.connect(self._on_relay_error)
        self.relay_worker.di2_triggered.connect(self._trigger_ocr)
        self.relay_worker.start()
        self.statusBar().showMessage(f"Connecting to relay at {host}:{port}...")

    def _disconnect_relay(self):
        if self.relay_worker:
            self.relay_worker.stop()
            self.relay_worker = None
        self._relay_connected = False
        self.relay_indicator.setText("  RELAY: Disconnected  ")
        self.relay_indicator.setObjectName("statusOff")
        self.relay_indicator.setStyleSheet(
            self.relay_indicator.styleSheet()
        )  # force refresh
        self.btn_connect_relay.setEnabled(True)
        self.btn_disconnect_relay.setEnabled(False)
        self.statusBar().showMessage("Relay disconnected")
        logger.info("Relay disconnected")

    @Slot(bool)
    def _on_relay_connected(self, connected: bool):
        self._relay_connected = connected
        if connected:
            self.relay_indicator.setText("  RELAY: Connected  ")
            self.relay_indicator.setObjectName("statusOn")
            self.btn_connect_relay.setEnabled(False)
            self.btn_disconnect_relay.setEnabled(True)
        else:
            self.relay_indicator.setText("  RELAY: Failed  ")
            self.relay_indicator.setObjectName("statusOff")
        self.relay_indicator.style().unpolish(self.relay_indicator)
        self.relay_indicator.style().polish(self.relay_indicator)

    @Slot(str)
    def _on_relay_error(self, error: str):
        self.statusBar().showMessage(f"Relay Error: {error}")
        logger.error(f"Relay: {error}")

    # ── Log Tab Actions ──────────────────────────────────────
    def _refresh_log(self):
        new_entries = self._log_handler.get_new_entries()
        if new_entries:
            for entry in new_entries:
                # Color-code log entries
                if "[ERROR]" in entry:
                    colored = f'<span style="color:#f38ba8;">{entry}</span>'
                elif "[WARNING]" in entry:
                    colored = f'<span style="color:#f9e2af;">{entry}</span>'
                elif "[INFO]" in entry:
                    colored = f'<span style="color:#a6e3a1;">{entry}</span>'
                else:
                    colored = entry
                self.log_text.append(colored)

    def _clear_log(self):
        self.log_text.clear()
        self._log_handler.clear()
        logger.info("Log cleared")

    def _export_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", str(self._app_dir / "log_export.txt"),
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.toPlainText())
                self.statusBar().showMessage(f"Log exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    # ── Cleanup ──────────────────────────────────────────────
    def closeEvent(self, event):
        logger.info("Application closing...")

        self._timer.stop()
        self._log_timer.stop()

        if self._capture is not None:
            self._capture.release()

        self.ocr_worker.stop()

        if self.relay_worker:
            self.relay_worker.stop()

        event.accept()


# ──────────────────────────────────────────────────────────────
# Custom Log Handler to capture logs for UI display
# ──────────────────────────────────────────────────────────────
class LogHandler(logging.Handler):
    """Buffers log entries for display in the Log tab."""
    def __init__(self):
        super().__init__()
        self._entries = []
        self._read_index = 0
        self._lock = QMutex()

    def emit(self, record):
        msg = self.format(record)
        with QMutexLocker(self._lock):
            self._entries.append(msg)

    def get_new_entries(self) -> list:
        with QMutexLocker(self._lock):
            new = self._entries[self._read_index:]
            self._read_index = len(self._entries)
            return new

    def clear(self):
        with QMutexLocker(self._lock):
            self._entries.clear()
            self._read_index = 0


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def main():
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AGC LOGO MARK")
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "logo.ico")))
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
