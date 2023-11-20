import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
haar_data = cv2.CascadeClassifier("C:\\Users\\sanja\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

#THE FOLLOWING TO BIG BLOCKS OF CODE NEEDS TO BE EXECUTED ONLY ONCE SINCE IT IS FOR COLLECTING IMAGE FOR TRAINING THE MODEL.
#BUT YES, FURTHER IMAGES CAN BE ADDED WHICH WILL INCREASE THE ACCURACY OF THE OUTPUT

# def withmask():
#     data_withmask = []
#     while True:
#         flag, img = cap.read()
#         if flag:
#             faces = haar_data.detectMultiScale(img)
#             for x,y,w,h in faces:
#                 cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
#                 face = img[y:y+h, x:x+w, :]
#                 face = cv2.resize(face, (50,50))
#                 print(len(data_withmask))
#                 if len(data_withmask)<400:
#                     data_withmask.append(face)
#             cv2.imshow("Result", img)
#             #27 is the ASCII value of escape
#             if cv2.waitKey(2) == 27 or len(data_withmask)>=400:
#                 break
#     cap.release()
#     cv2.destroyAllWindows()
#     np.save("With_Mask.npy", data_withmask)

def withoutmask():
    data_withoutmask = []
    while True:
        flag, img = cap.read()
        if flag:
            faces = haar_data.detectMultiScale(img)
            for x,y,w,h in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
                face = img[y:y+h, x:x+w, :]
                face = cv2.resize(face, (50,50))
                print(len(data_withoutmask))
                if len(data_withoutmask)<400:
                    data_withoutmask.append(face)
            cv2.imshow("Result", img)
            #27 is the ASCII value of escape
            if cv2.waitKey(2) == 27 or len(data_withoutmask)>=400:
                break

    cap.release()
    cv2.destroyAllWindows()   
    np.save("Without_Mask.npy", data_withoutmask)

withoutmask()
# withmask()
