# AGC LOGO MARK - Real-Time OCR Inspector

Real-time OCR inspection application built with **PySide6** and **EasyOCR**, optimized for **NVIDIA Jetson Orin Nano** with CUDA GPU acceleration. Designed for manufacturing quality control — verifying text, logos, and markings on products via camera with hardware relay integration.

---

## Features

### Monitor Tab
- Live camera feed with real-time display
- **Trigger OCR** to capture and freeze a frame, then detect text
- Rotated bounding boxes with confidence-colored overlays on detected text
- Compare detected text against a loaded job model (JSON)
- Color-coded results: **green** = Match, **red** = Not Found
- Edit expected texts inline and save changes
- Image rotation slider for adjusting orientation
- Continuous OCR mode for ongoing inspection

### Job Change Tab
- Auto-scans directory for `.json` job files
- Click a job button → confirmation dialog → loads as current job
- **Right-click** a job button → delete with confirmation
- **Add New Job** button to create jobs via dialog
- **Load Job File** button searches by name (supports barcode input with `$` delimiter cleaning)

### Settings Tab
- Camera index selection
- Relay board IP/port configuration with connect/disconnect
- GPU (CUDA) toggle for OCR engine

### Log Tab
- Color-coded application log viewer (green=INFO, yellow=WARNING, red=ERROR)
- Export log to file
- Clear log

---

## Hardware Integration

### Relay Board (Waveshare Modbus POE Ethernet Relay)
- **Connection**: Ethernet TCP/IP (default: `192.168.1.200:502`)
- **DI2**: Triggers OCR processing (rising edge detection)
- **Relay CH4**: Turned ON when all texts match (auto-stops after 5 seconds)
- **Relay CH5**: Turned ON when any text is not found (CH4 turned OFF)
- Stop inspection turns off both CH4 and CH5

### Camera
- OpenCV video capture, configurable camera index
- Default resolution: 1280×720
- ~33 FPS display refresh rate

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   UI Thread   │     │  OCR Worker  │     │ Relay Worker │
│  (PySide6)   │◄───►│  (EasyOCR +  │     │  (Modbus     │
│  Camera Feed  │     │   CUDA GPU)  │     │   TCP/IP)    │
│  Results      │     │  Background  │     │  Background  │
│  Display      │     │  Thread      │     │  Thread      │
└──────────────┘     └──────────────┘     └──────────────┘
```

- **3 threads**: UI (camera + display), OCR worker (CUDA inference), Relay worker (DI polling)
- Frame freezes on OCR trigger, displays annotated results until stopped
- Rising-edge detection on DI2 prevents repeated triggers

---

## Project Structure

```
OCR_Mark/
├── main.py              # Main application (PySide6 + EasyOCR + CUDA)
├── Relay_B.py           # Waveshare Modbus relay board driver
├── requirements.txt     # Python dependencies
├── readme.md            # This file
├── summary.md           # Detailed project documentation
├── logo.ico             # Application icon
├── logs/                # Application log files (auto-created)
├── ocr_models/          # EasyOCR model cache (auto-created)
└── *.json               # Job/model files (dynamically created)
```

---

## Installation

### Jetson Orin Nano (Production)

1. **Flash JetPack 6.x** (includes CUDA, cuDNN, TensorRT)
2. Install PyTorch for Jetson from [NVIDIA](https://forums.developer.nvidia.com/t/pytorch-for-jetson/) — **do NOT** `pip install torch`
3. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
4. Run:
   ```bash
   python3 main.py
   ```

### Windows / Linux Desktop (Development)

```bash
pip install -r requirements.txt
python main.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PySide6 | Qt-based UI framework |
| EasyOCR | Text detection & recognition (CUDA-accelerated) |
| OpenCV | Video capture & image processing |
| NumPy | Array operations |
| PyTorch | Deep learning backend (pre-installed on Jetson) |

---

## Job File Format

Job files are simple JSON arrays of expected text strings:

```json
[
    "TOYOTA",
    "MADE IN JAPAN",
    "2025"
]
```

### Text Matching Logic
- **Exact match**: detected text equals expected text (case-insensitive)
- **Substring match**: expected text found within detected text, or vice versa
- **Combined match**: expected text found across all detected texts combined
- Any match type counts as **MATCH** (green)

### Barcode Input Cleaning
When loading a job via text input, barcode strings with `$` delimiters are automatically cleaned:
```
FOD11850100163$1SRG14R(BRK)-MM-4FIMXA-A7$15
→ extracts: SRG14R(BRK)-MM-4FIMXA-A7
```

---

## Usage

1. **Create a job**: Job Change tab → "Add New Job" → enter name and expected texts
2. **Load a job**: Click a job button (with confirmation) or use "Load Job File" in Monitor tab
3. **Inspect**: Click "Trigger OCR" or use relay DI2 hardware signal
4. **Review results**: Frozen frame shows annotated detections; results table shows match status
5. **Stop**: Click "Stop" or wait for auto-stop (5s after all-match relay trigger)

---

## License

Internal use — AGC Logo Mark inspection system.
