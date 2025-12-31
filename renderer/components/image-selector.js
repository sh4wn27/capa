/**
 * Image selector component - allows user to manually select/crop object
 * This works for ANY object, not just predefined classes
 */

class ImageSelector {
  constructor(imageElement, onSelect) {
    this.imageElement = imageElement
    this.onSelect = onSelect
    this.isSelecting = false
    this.startX = 0
    this.startY = 0
    this.selectionBox = null
    this.canvas = null
    this.ctx = null
    
    this.init()
  }

  init() {
    // Create overlay canvas for selection
    const container = this.imageElement.parentElement
    this.canvas = document.createElement('canvas')
    this.canvas.style.position = 'absolute'
    this.canvas.style.top = '0'
    this.canvas.style.left = '0'
    this.canvas.style.pointerEvents = 'none'
    this.canvas.style.border = '2px dashed #667eea'
    container.style.position = 'relative'
    container.appendChild(this.canvas)

    this.ctx = this.canvas.getContext('2d')
    this.setupEventListeners()
  }

  setupEventListeners() {
    this.imageElement.addEventListener('mousedown', this.onMouseDown.bind(this))
    this.imageElement.addEventListener('mousemove', this.onMouseMove.bind(this))
    this.imageElement.addEventListener('mouseup', this.onMouseUp.bind(this))
    this.imageElement.style.cursor = 'crosshair'
  }

  onMouseDown(e) {
    this.isSelecting = true
    const rect = this.imageElement.getBoundingClientRect()
    this.startX = e.clientX - rect.left
    this.startY = e.clientY - rect.top
    
    // Resize canvas to match image
    this.canvas.width = this.imageElement.offsetWidth
    this.canvas.height = this.imageElement.offsetHeight
    this.canvas.style.width = this.imageElement.offsetWidth + 'px'
    this.canvas.style.height = this.imageElement.offsetHeight + 'px'
  }

  onMouseMove(e) {
    if (!this.isSelecting) return

    const rect = this.imageElement.getBoundingClientRect()
    const currentX = e.clientX - rect.left
    const currentY = e.clientY - rect.top

    const width = currentX - this.startX
    const height = currentY - this.startY

    // Clear and redraw selection box
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.strokeStyle = '#667eea'
    this.ctx.lineWidth = 2
    this.ctx.setLineDash([5, 5])
    this.ctx.strokeRect(this.startX, this.startY, width, height)

    // Store selection box
    this.selectionBox = {
      x: Math.min(this.startX, currentX),
      y: Math.min(this.startY, currentY),
      width: Math.abs(width),
      height: Math.abs(height)
    }
  }

  onMouseUp(e) {
    if (!this.isSelecting) return
    this.isSelecting = false

    if (this.selectionBox && this.selectionBox.width > 10 && this.selectionBox.height > 10) {
      // Get image coordinates (account for image scaling)
      const imgRect = this.imageElement.getBoundingClientRect()
      const scaleX = this.imageElement.naturalWidth / imgRect.width
      const scaleY = this.imageElement.naturalHeight / imgRect.height

      const selection = {
        x: this.selectionBox.x * scaleX,
        y: this.selectionBox.y * scaleY,
        width: this.selectionBox.width * scaleX,
        height: this.selectionBox.height * scaleY
      }

      this.onSelect(selection)
    } else {
      // Clear selection if too small
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    }
  }

  clear() {
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    }
    this.selectionBox = null
  }

  destroy() {
    if (this.canvas && this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }
}

module.exports = ImageSelector

