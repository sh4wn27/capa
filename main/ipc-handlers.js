const { ipcMain, dialog } = require('electron/main')
const fs = require('fs').promises
const path = require('node:path')
const { getDetector } = require('../ml/object-detection/detector')

/**
 * Sets up all IPC handlers for communication between renderer and main process
 */
function setupIpcHandlers() {
  // Image capture handlers
  ipcMain.handle('capture-screenshot', async () => {
    // TODO: Implement screenshot capture
    return { success: false, message: 'Not implemented yet' }
  })

  ipcMain.handle('select-image-file', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [
        { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }
      ]
    })

    if (result.canceled) {
      return { success: false, canceled: true }
    }

    try {
      const filePath = result.filePaths[0]
      const imageData = await fs.readFile(filePath)
      const base64 = imageData.toString('base64')
      
      return {
        success: true,
        filePath,
        imageData: `data:image/${path.extname(filePath).slice(1)};base64,${base64}`
      }
    } catch (error) {
      return { success: false, error: error.message }
    }
  })

  // Object detection handler
  ipcMain.handle('detect-object', async (event, imageData) => {
    try {
      console.log('Starting object detection...')
      const detector = getDetector()
      const result = await detector.detect(imageData)
      console.log('Detection result:', result)
      return result
    } catch (error) {
      console.error('Detection error in IPC handler:', error)
      return {
        success: false,
        error: error.message,
        detectedObject: 'unknown',
        confidence: 0,
        boundingBox: null
      }
    }
  })

  // 3D generation handler
  ipcMain.handle('generate-3d', async (event, imageData, objectInfo) => {
    // TODO: Integrate 3D generation pipeline
    return {
      success: false,
      message: '3D generation not implemented yet'
    }
  })

  // Export handler
  ipcMain.handle('export-model', async (event, modelData, format) => {
    const result = await dialog.showSaveDialog({
      filters: [
        { name: 'STL Files', extensions: ['stl'] },
        { name: 'OBJ Files', extensions: ['obj'] },
        { name: 'GLB Files', extensions: ['glb'] }
      ],
      defaultPath: `model.${format}`
    })

    if (result.canceled) {
      return { success: false, canceled: true }
    }

    // TODO: Implement actual export
    return {
      success: false,
      message: 'Export not implemented yet',
      filePath: result.filePath
    }
  })

  // Legacy ping handler (for testing)
  ipcMain.handle('ping', () => 'pong')
}

module.exports = { setupIpcHandlers }

