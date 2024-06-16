import cv2
from mtcnn.mtcnn import MTCNN
from utils.time_lag import TimeLag

#? Initializing the Face Detector
detector = MTCNN()

#? Loading the video
vid = cv2.VideoCapture("../videos/demo2.mp4")

#? Initializing the TimeLag object
time_lag = TimeLag()


while True:
    ret, frame = vid.read()

    if frame is None:
        break

    if frame.shape[0] > 1000:
        frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))

    time_lag.register_time()

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('MTCNN Face Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


#? Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()

print(f"\nAvg. Detection Time: {time_lag.get_time_lag()} seconds")