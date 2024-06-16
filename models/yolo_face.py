import cv2
import numpy as np

class YOLOFace:
    def __init__(self, weights, config, confidence=0.5):
        self.model = cv2.dnn.readNet(weights, config)
        self.confidence = confidence

        self.layer_names = self.model.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]


    def detect(self, frame):
        h, w = frame.shape[:2]

        #* Convert the image to blob
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

        #* Detect the faces
        self.model.setInput(blob)
        detections = self.model.forward(self.output_layers)

        confidences = []
        boxes = []

        #* Loop over the detections to get the coordinates for bounding boxes
        for faces in detections:
            detection = faces[:, -1] > self.confidence
            faces = faces[detection]

            for face in faces:
                center_x = int(face[0] * w)
                center_y = int(face[1] * h)
                width = int(face[2] * w)
                height = int(face[3] * h)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(face[-1]))

        #* Apply Non-Maximum Suppression to get the best bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, 0.4)

        return boxes, indexes


    def detect_along_optical_flow(self, frame):
        boxes, indexes = self.detect(frame)

        points = []
        dims = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, width, height = boxes[i]

                center = np.array([x + width // 2, y + height // 2], dtype=np.float32)
                points.append(center)
                dims.append([width, height])

        points = np.array(points).reshape(-1, 1, 2)

        return points, dims