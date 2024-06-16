import cv2
import numpy as np
from utils.time_lag import TimeLag

#* GLOBAL VARIABLES
CONFIDENCE = 0.5

#? Loading the Single Shot Detection model
model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")

#? Loading the video
vid = cv2.VideoCapture("../videos/demo.mp4")

#? Initializing the TimeLag object
time_lag = TimeLag()


while True:
    _, frame = vid.read()

    if frame is None:
        break

    if frame.shape[0] > 1000:
        frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    
    time_lag.register_time()

    (h, w) = frame.shape[:2]

    #? Resize the frame to 300x300
    img = cv2.resize(frame, (300, 300))

    #? Convert the image to blob
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    #? Detect the faces
    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break


#? Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()

print(f"\nAvg. Detection Time: {time_lag.get_time_lag()} seconds")