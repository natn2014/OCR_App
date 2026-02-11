# AGC LOGO MARK - Real-Time OCR Application

## Project Overview
AGC LOGO MARK is a real-time Optical Character Recognition (OCR) application built with **PySide6** that integrates computer vision with hardware relay control. It's designed to inspect and verify text/logos in images using EasyOCR with **CUDA GPU acceleration** (optimized for NVIDIA Jetson Orin Nano), making it suitable for quality control and inspection systems.

## Key Features

### 1. **Monitor Tab**
- **Real-time Video Feed**: Continuous camera stream display
- **OCR Processing**: Detects and extracts text from video frames
- **Visual Feedback**: Rotate Image to keep text is Horizontal and Draws rectangles and labels around detected text in the video
- **Model Data Management**: Load JSON files containing reference texts to compare against
- **Result Comparison**: Displays match status (Match, Partial Match, Not Found) with color coding:
  - Green: Exact match found
  - Yellow: Partial match found
  - Red: No match found
- **Edit & Save**: Ability to modify comparison texts and save changes

### 2. **Job Change Tab**
- **Dynamic Job Selection**: Automatically scans for all `.json` files in the current directory
- **Button Interface**: Creates individual buttons for each JSON file with the filename as the label
  - Example: `Test1.json` → displays as `Test1` button / for 1 row can contain 2 button.

- **Quick Job Loading**: Click any button to instantly load that job's data into the Monitor tab
- **Refresh Functionality**: "Refresh Jobs" button to reload the file list
- **Add New Job**: Click on "Add New Job" will show dialog that waiting for input data for detect and name of Json file.
- **Scrollable Layout**: Handles large numbers of job files efficiently

### 3. **Setting Tab**
- **Model Management**: Create and manage new OCR comparison models
- **Edit & Save Models**: Add/edit text entries and save them as new JSON files
- **Dynamic Model Creation**: Name models and store them for later use

### 4. **Log Tab**
- **Event Logging**: Placeholder for system logs and activity tracking
- **Future Expansion**: Ready for logging OCR results and system events

## Technical Architecture

### Core Components

#### **EasyOCR Reader**
- Detects and recognizes English text in images
- Returns bounding boxes and confidence scores for detected text
- Used for real-time inspection

#### **Video Capture**
- Integrates OpenCV for camera input
- Runs at 30ms update interval (~33 FPS)
- Supports dynamic resolution from connected camera

#### **Relay Integration (Waveshare Modbus POE)**
- Connects to relay hardware board (IP: 192.168.1.200)
- Monitors digital inputs (DI1, DI2) for hardware signals
- **DI1**: Triggers OCR processing
- **DI2**: Stops inspection process
- Uses separate worker thread for relay monitoring

#### **JSON File Management**
- Stores comparison texts as JSON arrays
- Supports loading, editing, and saving model data
- Enables easy job switching without code changes

### Data Flow
```
Camera Input 
    ↓
Video Feed Display (Monitor Tab)
    ↓
EasyOCR Processing
    ↓
Text Detection & Bounding Box Drawing
    ↓
Compare with Model Data (JSON)
    ↓
Display Results with Color Coding
    ↓
Save/Edit Results
```

## Hardware Integration

### Relay Board Control
- **Model**: Waveshare Modbus POE Ethernet Relay
- **Connection**: Ethernet (TCP/IP)
- **Purpose**: Synchronize OCR processing with external inspection systems
- **Features**:
  - 8 relay channels for device control
  - Digital inputs for hardware triggers
  - CRC-based Modbus protocol communication

## File Structure
```
OCR_Mark/
├── main.py              # Main application file (PySide6 + EasyOCR + CUDA)
├── Relay_B.py           # Relay board communication driver
├── requirements.txt     # Python dependencies
├── summary.md           # Project documentation
├── logs/                # Application log files (auto-created)
├── ocr_models/          # EasyOCR model cache (auto-created)
└── [model files].json   # Dynamically created job files
```

## Example Workflow

1. **Create Models**: Use Setting tab to create `Test1.json` containing texts like ["LOGO", "SERIAL", "DATE"]
2. **Switch Jobs**: Use Job Change tab to click "Test1" button to load that model
3. **Run Inspection**: Monitor tab shows camera feed; click "Trigger OCR" or use relay input (DI1)
4. **View Results**: 
   - Green cells = detected all required texts
   - Yellow cells = partially matched
   - Red cells = missing texts
5. **Edit & Save**: Modify results and save changes back to JSON

## Dependencies
- **PySide6**: UI framework (dark theme, modern design)
- **EasyOCR**: Text detection and recognition (CUDA-accelerated via PyTorch)
- **OpenCV (cv2)**: Video capture and image processing
- **PyTorch**: Deep learning backend (pre-installed on Jetson via JetPack)
- **Python 3.8+**: Runtime environment

## Jetson Orin Nano Setup
1. Flash JetPack 6.x (includes CUDA, cuDNN, TensorRT)
2. Install PyTorch for Jetson from NVIDIA (do NOT pip install torch)
3. `pip3 install -r requirements.txt`
4. `python3 main.py`

## Use Cases
- Manufacturing quality control (logo/text verification)
- Document inspection systems
- Product marking verification
- Automated visual inspection lines
- Logo detection and validation

## Future Enhancements
- Multi-language OCR support
- Machine learning-based text validation
- Database integration for results storage
- Network-based job management
- Report generation and statistics
- Advanced image preprocessing filters
