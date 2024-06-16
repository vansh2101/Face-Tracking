import cv2
from utils.time_lag import TimeLag
from models.yolo_face import YOLOFace

#* GLOBAL VARIABLES
CONFIDENCE = 0.5

#? Loading the YOLO model
model = YOLOFace("weights/yolov3-wider_16000.weights", "weights/yolov3-face.cfg", CONFIDENCE)

#? Loading the video
vid = cv2.VideoCapture("videos/demo.mp4")

#? Initializing the TimeLag object
time_lag = TimeLag()


while True:
    _, frame = vid.read()

    if frame is None:
        break

    if frame.shape[0] > 1000:
        frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    
    time_lag.register_time()

    boxes, indexes = model.detect(frame)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, wid, hei = boxes[i]

            cv2.rectangle(frame, (x, y), (x + wid, y + hei), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


#? Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()

print(f"\nAvg. Detection Time: {time_lag.get_time_lag()} seconds")