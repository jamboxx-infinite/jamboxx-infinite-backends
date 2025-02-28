# Jamboxx Infinite Backends

A FastAPI-based backend service for DDSP-SVC voice conversion and audio processing.

## Features

- Real-time voice conversion using DDSP-SVC
- Multiple speaker support
- Pitch adjustment capabilities
- RESTful API endpoints
- Async processing
- Docker support
- Comprehensive error handling

## Requirements

- Windows 10/11 (64-bit)
- Visual C++ Redistributable 2019 or later
- CUDA-compatible GPU (recommended)
- FFmpeg

## Installation

### Method 1: Using Pre-compiled Executable (Windows Only)

1. Download the latest release from the releases page
2. Extract the archive to your desired location
3. Run `start.bat` in the extracted folder

### Method 2: From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jamboxx-infinite-backends.git
cd jamboxx-infinite-backends
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

## Usage

### Running the Server

Using pre-compiled executable:
```batch
cd dist\main.dist
start.bat
```

Production mode:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Building from Source

To compile the application into a standalone executable:

```batch
cd scripts
build.bat
```

The compiled executable and required files will be available in the `dist/main.dist` directory.

### API Endpoints

#### Health Check
```
GET /ping
```

#### Voice Conversion
```
POST /api/v1/voiceConvert
```

Request body:
```json
{
    "audio_data": "base64_encoded_audio",
    "speaker_id": "speaker1",
    "pitch_shift": 0.0,
    "sample_rate": 44100
}
```

#### List Available Models
```
GET /api/v1/models
```

#### Update Model
```
POST /api/v1/updateModel
```

## Development

### Project Structure
```
jamboxx_infinite_backends/
├── app/
│   ├── main.py
│   ├── routers/
│   ├── services/
│   ├── core/
│   ├── utils/
│   └── schemas/
├── scripts/
│   ├── build.bat
│   └── download_models.py
├── dist/
│   └── main.dist/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Building Notes

- Compilation requires approximately 2GB of disk space
- First startup may take longer than subsequent launches
- Compiled version includes all necessary dependencies
- Target system must have Visual C++ Redistributable 2019 or later installed

## Acknowledgments

- DDSP-SVC project
- FastAPI framework
- PyTorch community
