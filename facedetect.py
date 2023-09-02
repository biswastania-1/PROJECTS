import cv2
import os

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath("E:\cv\haarcascade_frontalface_default.xml"))
cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")

alg = cascade_path
haar_cascade = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        break
    
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayimg, 1.3, 4)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
