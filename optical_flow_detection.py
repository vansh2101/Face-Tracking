import cv2
import numpy as np
from utils.time_lag import TimeLag
from models.yolo_face import YOLOFace

#* GLOBAL VARIABLES
CONFIDENCE = 0.5
skip = 10
count = 0
points = []
prev_gray = None

#? Loading the YOLO model
model = YOLOFace("weights/yolov3-wider_16000.weights", "weights/yolov3-face.cfg", CONFIDENCE)

#? Setting the parameters for Optical Flow
optical_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

    count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if count % skip == 0:
        points, dims = model.detect_along_optical_flow(frame)

        points = np.array(points).reshape(-1, 1, 2)

    else:
        #* Apply Optical Flow to track the faces
        if len(points) > 0 and prev_gray is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **optical_params)
            points = new_points[status == 1].reshape(-1, 1, 2)


    #* Draw the bounding boxes on the frame
    for index in range(len(points)):
        x, y = points[index].ravel()
        w, h = dims[index]

        cv2.rectangle(frame, (int(x - w//2), int(y - h//2)), (int(x + w//2), int(y + h//2)), (0, 255, 0), 2)

    cv2.imshow("Face Tracking System", frame)

    prev_gray = gray.copy()

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


#? Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()

print(f"\nAvg. Detection Time: {time_lag.get_time_lag()} seconds")