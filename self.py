# Importing Libraries
import cv2

# Loading Cascade
f = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
e = cv2.CascadeClassifier('haarcascade_eye.xml')
s = cv2.CascadeClassifier('haarcascade_smile.xml')

# Making function for recognising smile
def detect(gray, frame):
    faces = f.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = e.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = s.detectMultiScale(roi_gray, 1.7, 25)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

# Doing Smile Recognition with webcam
v  = cv2.VideoCapture('sample2.mp4')
while(True):
    _,frame = v.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
v.release()
cv2.destroyAllWindows()

    