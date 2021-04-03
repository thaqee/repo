import cv2
import playsound

cap = cv2.VideoCapture(0)

faceModel = cv2.CascadeClassifier("ai_face_brain.xml")

while True:

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Gray

	faces = faceModel.detectMultiScale(gray,1.2,5)
	
	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,255), 2)
		cv2.putText(frame, 'Face', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
		frame[y:y+h,x:x+w] = cv2.medianBlur(frame[y:y+h,x:x+w],35)
	

	number_of_faces = len(faces)

	if number_of_faces >=2:
		playsound.playsound("song.mp3")

	
	cv2.imshow("Frame", frame)
	cv2.imshow("Frame_2", gray)

	if cv2.waitKey(1)==ord('q'):
		break

cap.release()
cv2.destoryAllWindows()