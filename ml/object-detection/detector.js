const tf = require('@tensorflow/tfjs-node')
const cocoSsd = require('@tensorflow-models/coco-ssd')
const sharp = require('sharp')
const fs = require('fs').promises

/**
 * Object Detection Service using COCO-SSD
 * Detects objects in images and returns the most confident detection
 */
class ObjectDetector {
  constructor() {
    this.model = null
    this.isLoading = false
    this.modelPath = null
  }

  /**
   * Load the COCO-SSD model (lazy loading)
   */
  async loadModel() {
    if (this.model) {
      return this.model
    }

    if (this.isLoading) {
      // Wait for ongoing load
      while (this.isLoading) {
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      return this.model
    }

    this.isLoading = true
    try {
      console.log('Loading COCO-SSD model...')
      this.model = await cocoSsd.load({
        base: 'mobilenet_v2', // Faster, good for MVP
        // base: 'lite_mobilenet_v2' // Even faster but less accurate
      })
      console.log('Model loaded successfully')
      this.isLoading = false
      return this.model
    } catch (error) {
      this.isLoading = false
      console.error('Error loading model:', error)
      throw error
    }
  }

  /**
   * Convert base64 image to buffer
   */
  async base64ToBuffer(base64Data) {
    // Remove data URL prefix if present
    const base64 = base64Data.includes(',') 
      ? base64Data.split(',')[1] 
      : base64Data
    
    return Buffer.from(base64, 'base64')
  }

  /**
   * Preprocess image for TensorFlow
   * Resize and normalize if needed
   */
  async preprocessImage(imageBuffer) {
    // COCO-SSD works best with images around 300-640px
    // Sharp will handle resizing while maintaining aspect ratio
    const processed = await sharp(imageBuffer)
      .resize(640, 640, {
        fit: 'inside',
        withoutEnlargement: true
      })
      .toBuffer()
    
    return processed
  }

  /**
   * Detect objects in an image
   * @param {string} base64ImageData - Base64 encoded image (with or without data URL prefix)
   * @returns {Promise<Object>} Detection result with object name, confidence, and bounding box
   */
  async detect(base64ImageData) {
    try {
      // Load model if not already loaded
      const model = await this.loadModel()

      // Convert base64 to buffer
      const imageBuffer = await this.base64ToBuffer(base64ImageData)
      
      // Preprocess image
      const processedBuffer = await this.preprocessImage(imageBuffer)

      // Convert buffer to TensorFlow tensor
      // COCO-SSD can work with tensors directly in Node.js
      const imageTensor = tf.node.decodeImage(processedBuffer, 3) // 3 channels (RGB)

      try {
        // Run detection on tensor
        const predictions = await model.detect(imageTensor)

        // Dispose tensor to free memory
        imageTensor.dispose()

        if (!predictions || predictions.length === 0) {
          return {
            success: true,
            detectedObject: 'unknown',
            confidence: 0,
            boundingBox: null,
            allDetections: []
          }
        }

        // Get the most confident detection
        const topDetection = predictions.reduce((best, current) => 
          current.score > best.score ? current : best
        )

        // Format result
        return {
          success: true,
          detectedObject: topDetection.class,
          confidence: topDetection.score,
          boundingBox: {
            x: topDetection.bbox[0],
            y: topDetection.bbox[1],
            width: topDetection.bbox[2],
            height: topDetection.bbox[3]
          },
          allDetections: predictions.map(pred => ({
            class: pred.class,
            confidence: pred.score,
            bbox: {
              x: pred.bbox[0],
              y: pred.bbox[1],
              width: pred.bbox[2],
              height: pred.bbox[3]
            }
          }))
        }
      } catch (detectError) {
        // Dispose tensor on error
        if (imageTensor) {
          imageTensor.dispose()
        }
        throw detectError
      }
    } catch (error) {
      console.error('Detection error:', error)
      return {
        success: false,
        error: error.message,
        detectedObject: 'unknown',
        confidence: 0,
        boundingBox: null
      }
    }
  }

  /**
   * Get list of all detectable object classes
   */
  getDetectableClasses() {
    // COCO-SSD can detect 80 classes
    return [
      'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
      'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
      'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
      'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
      'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
      'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
      'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
      'toothbrush'
    ]
  }
}

// Singleton instance
let detectorInstance = null

/**
 * Get or create the detector instance
 */
function getDetector() {
  if (!detectorInstance) {
    detectorInstance = new ObjectDetector()
  }
  return detectorInstance
}

module.exports = { getDetector, ObjectDetector }

