import cv2
from utils.time_lag import TimeLag

#? Loading the haar cascade file
detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

#? Loading the Video
vid = cv2.VideoCapture('../videos/demo3.mp4')

#? Inititalize the class to calculate time lag
time_lag = TimeLag()


while True:
    _, frame = vid.read()

    if frame is None:
        break
    
    if frame.shape[0] > 1000:
        frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    
    time_lag.register_time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    print(faces)

    #? Draw a bounding box on all the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


#? Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()

print(f"\nAvg. Detection Time: {time_lag.get_time_lag()} seconds")