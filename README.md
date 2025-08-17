# YOLO Object Detection with OpenCV (C++)

This project demonstrates how to use a **YOLO model trained with Ultralytics** and exported to **ONNX format** for real-time object detection in C++ with **OpenCV DNN module**.  

The program loads a trained `best.onnx` model, reads class names from `classes.txt`, and performs object detection on video frames (`test.mp4`). Each detection is drawn with a bounding box, class label, and confidence score.  

### Key Features
- **Ultralytics-trained YOLO model (ONNX)**
- **OpenCV DNN inference in C++**
- Works with **YOLOv5 and YOLOv8 ONNX layouts**
- Real-time detection on video streams
- Non-Maximum Suppression (NMS) applied to filter results

