# Description: This project is to detect face and smile in an image or video
# Used libraries: OpenCV
# Author: Dhanush S
# Disclaimer: This detection for video will work only for inbuilt webcam
#              and not for external webcam and the detection for image will
#              work only for images with single face(Smile only, face detection works for multiple faces)
# Please note both xml files must be downloaded along with the code
import cv2 as cv

img = cv.imread("test.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade= cv.CascadeClassifier("h_smile.xml")
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
def face_detect(img, gray):
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img
def detect(gray_image, original_image):
    face = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for (x,y,w,h) in face:
        cv.rectangle(original_image, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray_image[y:y+h,x:x+w]
        roi_color = original_image[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smile:
            cv.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
    return original_image
print("1. Image\n2. Video")
choice = int(input("Enter your choice: "))
if choice == 1:
    face_detect(img, gray)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
elif choice == 2:
    video_capture = cv.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        canvas = detect(gray, frame)
        cv.imshow('Video',canvas)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv.destroyAllWindows()
else:
    print("Invalid choice")
