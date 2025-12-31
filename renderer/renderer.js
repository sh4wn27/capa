/**
 * Main renderer process logic for CAPA
 */

let currentImageData = null
let currentModel = null
let imageSelector = null
let selectedRegion = null

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners()
  updateStatus('Ready to capture')
})

function setupEventListeners() {
  // Image capture
  document.getElementById('btn-upload').addEventListener('click', handleImageUpload)
  document.getElementById('btn-screenshot').addEventListener('click', handleScreenshot)
  document.getElementById('btn-clear').addEventListener('click', clearImage)
  document.getElementById('btn-auto-detect').addEventListener('click', handleAutoDetect)

  // Object detection
  document.getElementById('btn-confirm').addEventListener('click', handleConfirmDetection)
  document.getElementById('btn-correct').addEventListener('click', handleCorrectDetection)
  document.getElementById('btn-show-all').addEventListener('click', toggleAllDetections)

  // 3D viewer controls
  document.getElementById('btn-rotate').addEventListener('click', () => {
    console.log('Rotate clicked')
  })
  document.getElementById('btn-scale').addEventListener('click', () => {
    console.log('Scale clicked')
  })
  document.getElementById('btn-reset').addEventListener('click', () => {
    console.log('Reset clicked')
  })

  // Export
  document.getElementById('btn-export-stl').addEventListener('click', () => handleExport('stl'))
  document.getElementById('btn-export-obj').addEventListener('click', () => handleExport('obj'))
  document.getElementById('btn-export-glb').addEventListener('click', () => handleExport('glb'))
}

async function handleImageUpload() {
  updateStatus('Selecting image...')
  
  try {
    const result = await window.electronAPI.selectImageFile()
    
    if (result.canceled) {
      updateStatus('Image selection canceled')
      return
    }

    if (result.success) {
      currentImageData = result.imageData
      displayImagePreview(result.imageData)
      updateStatus('Image loaded successfully')
      
      // Automatically proceed to detection
      setTimeout(() => {
        proceedToDetection()
      }, 500)
    } else {
      updateStatus(`Error: ${result.error}`)
    }
  } catch (error) {
    updateStatus(`Error: ${error.message}`)
    console.error('Upload error:', error)
  }
}

async function handleScreenshot() {
  updateStatus('Screenshot capture not implemented yet')
  // TODO: Implement screenshot capture
}

function displayImagePreview(imageData) {
  const preview = document.getElementById('image-preview')
  const img = document.getElementById('preview-img')
  img.src = imageData
  preview.classList.remove('hidden')
  
  // Initialize manual selection tool
  img.onload = () => {
    if (imageSelector) {
      imageSelector.destroy()
    }
    
    // Create simple selection handler (without external module for now)
    setupManualSelection(img)
  }
}

function setupManualSelection(img) {
  let isSelecting = false
  let startX = 0
  let startY = 0
  let selectionBox = null
  
  // Create overlay for selection
  const wrapper = img.parentElement
  wrapper.style.position = 'relative'
  
  // Remove existing overlay if any
  const existingOverlay = wrapper.querySelector('.selection-overlay')
  if (existingOverlay) {
    existingOverlay.remove()
  }
  
  const overlay = document.createElement('div')
  overlay.className = 'selection-overlay'
  overlay.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    border: 2px dashed black;
    background: rgba(0, 0, 0, 0.05);
    display: none;
    pointer-events: none;
  `
  wrapper.appendChild(overlay)
  
  img.style.cursor = 'crosshair'
  
  img.addEventListener('mousedown', (e) => {
    isSelecting = true
    const rect = img.getBoundingClientRect()
    startX = e.clientX - rect.left
    startY = e.clientY - rect.top
    overlay.style.display = 'block'
    overlay.style.left = startX + 'px'
    overlay.style.top = startY + 'px'
    overlay.style.width = '0px'
    overlay.style.height = '0px'
  })
  
  img.addEventListener('mousemove', (e) => {
    if (!isSelecting) return
    
    const rect = img.getBoundingClientRect()
    const currentX = e.clientX - rect.left
    const currentY = e.clientY - rect.top
    
    const width = currentX - startX
    const height = currentY - startY
    
    overlay.style.left = Math.min(startX, currentX) + 'px'
    overlay.style.top = Math.min(startY, currentY) + 'px'
    overlay.style.width = Math.abs(width) + 'px'
    overlay.style.height = Math.abs(height) + 'px'
    
    selectionBox = {
      x: Math.min(startX, currentX),
      y: Math.min(startY, currentY),
      width: Math.abs(width),
      height: Math.abs(height)
    }
  })
  
  img.addEventListener('mouseup', (e) => {
    if (!isSelecting) return
    isSelecting = false
    
    if (selectionBox && selectionBox.width > 10 && selectionBox.height > 10) {
      // Convert to image coordinates
      const scaleX = img.naturalWidth / img.offsetWidth
      const scaleY = img.naturalHeight / img.offsetHeight
      
      selectedRegion = {
        x: selectionBox.x * scaleX,
        y: selectionBox.y * scaleY,
        width: selectionBox.width * scaleX,
        height: selectionBox.height * scaleY
      }
      
      updateStatus('Object selected! Click "Generate 3D" or use Auto-Detect')
      
      // Show button to proceed
      const proceedBtn = document.createElement('button')
      proceedBtn.className = 'btn btn-primary'
      proceedBtn.textContent = 'Generate 3D from Selection'
      proceedBtn.style.marginTop = '10px'
      proceedBtn.onclick = () => {
        handleConfirmDetection()
      }
      
      const actions = document.querySelector('.image-actions')
      const existingBtn = actions.querySelector('.proceed-btn')
      if (existingBtn) existingBtn.remove()
      proceedBtn.className += ' proceed-btn'
      actions.appendChild(proceedBtn)
    } else {
      overlay.style.display = 'none'
      selectedRegion = null
    }
  })
}

function clearImage() {
  currentImageData = null
  selectedRegion = null
  document.getElementById('image-preview').classList.add('hidden')
  document.getElementById('preview-img').src = ''
  const overlay = document.querySelector('.selection-overlay')
  if (overlay) overlay.remove()
  const proceedBtn = document.querySelector('.proceed-btn')
  if (proceedBtn) proceedBtn.remove()
  showSection('capture-section')
  updateStatus('Image cleared')
}

async function handleAutoDetect() {
  if (!currentImageData) {
    updateStatus('Please upload an image first')
    return
  }
  
  updateStatus('Running AI detection...')
  showSection('detection-section')
  
  try {
    const result = await window.electronAPI.detectObject(currentImageData)
    
    if (result.success && result.detectedObject !== 'unknown') {
      displayDetectionResult(result)
      // If detection found something, use its bounding box
      if (result.boundingBox) {
        selectedRegion = result.boundingBox
      }
      updateStatus('Object detected!')
    } else {
      // Detection didn't find anything - suggest manual selection
      document.getElementById('detection-status').innerHTML = `
        <p>AI couldn't detect a known object.</p>
        <p>No problem. Go back and manually select the object by clicking and dragging on the image.</p>
        <p>This works for any object - Rubik's cube, custom items, etc.</p>
      `
      updateStatus('Use manual selection for custom objects')
    }
  } catch (error) {
    updateStatus(`Error: ${error.message}`)
    console.error('Auto-detect error:', error)
  }
}

async function proceedToDetection() {
  // Skip auto-detection if user already selected manually
  if (selectedRegion) {
    handleConfirmDetection()
    return
  }
  
  // Otherwise try auto-detection
  await handleAutoDetect()
}

function displayDetectionResult(result) {
  // Show primary detection
  const objectName = result.detectedObject || 'Unknown'
  const confidence = result.confidence || 0
  
  document.getElementById('detected-object').textContent = objectName
  document.getElementById('confidence').textContent = (confidence * 100).toFixed(1)
  
  // Show all detections if available
  if (result.allDetections && result.allDetections.length > 0) {
    const list = document.getElementById('detections-list')
    list.innerHTML = ''
    
    result.allDetections
      .sort((a, b) => b.confidence - a.confidence)
      .forEach((detection, index) => {
        const li = document.createElement('li')
        li.innerHTML = `
          <strong>${detection.class}</strong> 
          <span class="confidence-badge">${(detection.confidence * 100).toFixed(1)}%</span>
        `
        if (index === 0) {
          li.classList.add('primary')
        }
        list.appendChild(li)
      })
  }
  
  document.getElementById('detection-result').classList.remove('hidden')
  document.getElementById('detection-status').classList.add('hidden')
  
  // Show error if detection failed
  if (!result.success) {
    updateStatus(`Detection error: ${result.error || 'Unknown error'}`)
  }
}

let showingAllDetections = false

function toggleAllDetections() {
  const allDetectionsDiv = document.getElementById('all-detections')
  showingAllDetections = !showingAllDetections
  
  if (showingAllDetections) {
    allDetectionsDiv.classList.remove('hidden')
    document.getElementById('btn-show-all').textContent = 'Hide All'
  } else {
    allDetectionsDiv.classList.add('hidden')
    document.getElementById('btn-show-all').textContent = 'Show All'
  }
}

function handleConfirmDetection() {
  if (!selectedRegion && !currentImageData) {
    updateStatus('Please select an object first (click and drag on image)')
    return
  }
  
  updateStatus('Generating 3D model...')
  
  // Pass selected region to 3D generation
  const objectInfo = {
    region: selectedRegion,
    imageData: currentImageData
  }
  
  // TODO: Call 3D generation with objectInfo
  setTimeout(() => {
    showSection('viewer-section')
    initialize3DViewer()
    updateStatus('3D model ready')
  }, 1000)
}

function handleCorrectDetection() {
  // TODO: Show input field for manual correction
  alert('Manual correction coming soon!')
}

function initialize3DViewer() {
  // TODO: Initialize Three.js viewer
  const canvas = document.getElementById('canvas-3d')
  const ctx = canvas.getContext('2d')
  
  // Placeholder: Draw a simple message
  ctx.fillStyle = '#667eea'
  ctx.font = '24px Arial'
  ctx.textAlign = 'center'
  ctx.fillText('3D Viewer Coming Soon', canvas.width / 2, canvas.height / 2)
  
  updateStatus('3D viewer initialized (placeholder)')
}

async function handleExport(format) {
  updateStatus(`Exporting as ${format.toUpperCase()}...`)
  
  try {
    const result = await window.electronAPI.exportModel(currentModel, format)
    
    if (result.canceled) {
      updateStatus('Export canceled')
      return
    }

    if (result.success) {
      updateStatus(`Exported to ${result.filePath}`)
    } else {
      updateStatus(`Export failed: ${result.message}`)
    }
  } catch (error) {
    updateStatus(`Error: ${error.message}`)
    console.error('Export error:', error)
  }
}

function showSection(sectionId) {
  // Hide all sections
  document.querySelectorAll('.step-section').forEach(section => {
    section.classList.remove('active')
    section.classList.add('hidden')
  })
  
  // Show target section
  const target = document.getElementById(sectionId)
  if (target) {
    target.classList.add('active')
    target.classList.remove('hidden')
  }
}

function updateStatus(message) {
  document.getElementById('status-text').textContent = message
  console.log(`[STATUS] ${message}`)
}

