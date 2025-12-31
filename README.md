# CAPA - Capture to 3D

**CAPA** is a desktop application that allows users to capture real-world objects from images and turn them into interactive 3D models that can be manipulated in real time and exported for CAD workflows.

## 🎯 Project Overview

CAPA bridges **2D visual input → 3D understanding → spatial interaction → CAD-ready output**, all from a single desktop interface.

### Core Features

- 📸 **Image Capture**: Upload images or take screenshots
- 🤖 **AI Object Detection**: Automatically identify objects in images
- 🎨 **3D Generation**: Convert 2D images into 3D models
- 🖱️ **Interactive Viewer**: Rotate, scale, and manipulate 3D models
- 📤 **CAD Export**: Export models in STL, OBJ, GLB formats
- ✋ **Gesture Control** (Coming Soon): Control 3D models with hand gestures
- 📽️ **Projection Mode** (Coming Soon): Project 3D models into physical space

## 🏗️ Architecture

```
capa/
├── main/                    # Electron main process
│   ├── main.js             # App entry, window management
│   └── ipc-handlers.js     # IPC handlers for ML operations
├── renderer/                # Frontend (UI)
│   ├── index.html
│   ├── renderer.js         # Main UI logic
│   ├── components/         # UI components (to be added)
│   └── styles/
│       └── main.css
├── preload.js              # Context bridge (security)
├── ml/                     # ML/3D processing
│   ├── object-detection/   # Object recognition
│   ├── depth-estimation/   # 2D → 3D conversion
│   └── mesh-generation/    # Mesh creation/refinement
└── utils/                  # Shared utilities
    └── export.js           # CAD export functions
```

## 🚀 Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm

### Installation

```bash
npm install
```

### Running the App

```bash
npm start
```

For development with DevTools:
```bash
npm start -- --dev
```

## 📋 Development Roadmap

### Phase 1: MVP (Current)
- [x] Project structure and architecture
- [x] Basic UI with image upload
- [x] IPC communication setup
- [ ] Object detection integration
- [ ] Basic 3D viewer with Three.js
- [ ] Simple 3D mesh generation
- [ ] Export functionality (STL/OBJ)

### Phase 2: Enhanced 3D Quality
- [ ] Improved reconstruction algorithms
- [ ] Multi-view input support
- [ ] Mesh refinement tools
- [ ] User editing capabilities

### Phase 3: Gesture Control
- [ ] MediaPipe Hands integration
- [ ] Gesture-to-transformation mapping
- [ ] Camera access

### Phase 4: Advanced Features
- [ ] Projector calibration
- [ ] Real-time projection
- [ ] Advanced CAD formats (STEP/IGES)

## 🛠️ Tech Stack

- **Electron**: Desktop application framework
- **Three.js**: 3D rendering and manipulation
- **Node.js**: Backend processing
- **MediaPipe** (planned): Hand gesture tracking
- **TensorFlow.js / ONNX.js** (planned): ML models

## 📝 License

ISC

## 👤 Author

sh4wn

