import cv2
#อ่านวีดีโอเข้ามาทำงาน
cap = cv2.VideoCapture("Detect/Detect/Mark.mp4")
face_casecade = cv2.CascadeClassifier("Detect/Detect/faces.xml")

#แสดงวีดีโอ
while (cap.isOpened()):
    check,frame = cap.read()
    if check == True:
        #แปลงภาพสี -> grayscale
        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #จำแนกใบหน้า
        face_detect = face_casecade.detectMultiScale(gray_img,1.3,5)
        #บอกพื้นที่ที่เจอใบหน้า
        for (x,y,w,h) in face_detect:
            #เซ็นเซอร์ใบหน้า
            frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h,x:x+w],(50,50))
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=5)
            #แสดงเฟรมในวีดีโอ
            cv2.imshow("Output",frame)
        #กดปุ่ม e เพื่อปิดหน้าต่าง
        if cv2.waitKey(1) & 0xFF==ord("e"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()