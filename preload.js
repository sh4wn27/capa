const { contextBridge, ipcRenderer } = require('electron')

/**
 * Exposes safe APIs to the renderer process
 */
contextBridge.exposeInMainWorld('electronAPI', {
  // Image capture
  captureScreenshot: () => ipcRenderer.invoke('capture-screenshot'),
  selectImageFile: () => ipcRenderer.invoke('select-image-file'),

  // Object detection
  detectObject: (imageData) => ipcRenderer.invoke('detect-object', imageData),

  // 3D generation
  generate3D: (imageData, objectInfo) => 
    ipcRenderer.invoke('generate-3d', imageData, objectInfo),

  // Export
  exportModel: (modelData, format) => 
    ipcRenderer.invoke('export-model', modelData, format),

  // Utility
  ping: () => ipcRenderer.invoke('ping')
})
