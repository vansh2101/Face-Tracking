import cv2
import mediapipe as mp
from utils.time_lag import TimeLag

#? Initialize the mediapipe face detection module
mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection(min_detection_confidence=0.5, model_selection=1)
draw = mp.solutions.drawing_utils

#? Loading the video
vid = cv2.VideoCapture('../videos/demo2.mp4')

#? Inititalize the class to calculate time lag
time_lag = TimeLag()


while True:
    _, frame = vid.read()

    if frame is None:
        break
    
    if frame.shape[0] > 1000:
        frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    
    time_lag.register_time()

    results = face.process(frame)

    #? Draw a bounding box on all the detected faces
    if results.detections:
        for detection in results.detections:
            draw.draw_detection(frame, detection)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break


#? Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()

print(f"\nAvg. Detection Time: {time_lag.get_time_lag()} seconds")