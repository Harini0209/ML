import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('kalam.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.1, 4):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()