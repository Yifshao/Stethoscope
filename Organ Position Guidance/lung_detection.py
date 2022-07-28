import mediapipe
import cv2
import numpy as np
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

cap = cv2.VideoCapture(0)
detect_heart = True
with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:	
	while cap.isOpened():
		ret, frame = cap.read()
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = holistic.process(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		SHOULDERDEFINE = 165
		keypoints = []
		sx1 = 0
		sx2 = 0
		sy1 = 0
		sy2 = 0
		shoulderlen = 0
		radius = 0
		px1 = 0 
		py1 = 0
		px2 = 0
		py2 = 0
		px3 = 0
		py3 = 0
		px4 = 0
		py4 = 0
		font = cv2.FONT_HERSHEY_SIMPLEX
		hearttest = False
		aortic = False
		pulmonary = False
		tricuspid = False
		mitral = False
		if results.pose_landmarks and detect_heart:
			shoulderleft = results.pose_landmarks.landmark[11]
			shoulderright = results.pose_landmarks.landmark[12]
			#heartx = (shoulderleft.x*3/4) + shoulderright.x/4
			#hearty = (shoulderleft.y + shoulderright.y)/2 + (3/4)*(abs(shoulderright.x-shoulderleft.x))
			shoulderlen = abs(shoulderright.x-shoulderleft.x)
			rightx1 = shoulderright.x + (70/SHOULDERDEFINE)*shoulderlen
			righty1 = (shoulderleft.y + shoulderright.y)/2 + (8/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx2 = shoulderright.x + (55/SHOULDERDEFINE)*shoulderlen
			righty2 = (shoulderleft.y + shoulderright.y)/2 + (18/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx3 = shoulderright.x + (41/SHOULDERDEFINE)*shoulderlen
			righty3 = (shoulderleft.y + shoulderright.y)/2 + (38/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx4 = shoulderright.x + (30/SHOULDERDEFINE)*shoulderlen
			righty4 = (shoulderleft.y + shoulderright.y)/2 + (60/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx5 = shoulderright.x + (23/SHOULDERDEFINE)*shoulderlen
			righty5 = (shoulderleft.y + shoulderright.y)/2 + (77/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx6 = shoulderright.x + (18/SHOULDERDEFINE)*shoulderlen
			righty6 = (shoulderleft.y + shoulderright.y)/2 + (94/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx7 = shoulderright.x + (15/SHOULDERDEFINE)*shoulderlen
			righty7 = (shoulderleft.y + shoulderright.y)/2 + (108/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx8 = shoulderright.x + (16/SHOULDERDEFINE)*shoulderlen
			righty8 = (shoulderleft.y + shoulderright.y)/2 + (132/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx9 = shoulderright.x + (17/SHOULDERDEFINE)*shoulderlen
			righty9 = (shoulderleft.y + shoulderright.y)/2 + (154/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx10 = shoulderright.x + (30/SHOULDERDEFINE)*shoulderlen
			righty10 = (shoulderleft.y + shoulderright.y)/2 + (153/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx11 = shoulderright.x + (50/SHOULDERDEFINE)*shoulderlen
			righty11 = (shoulderleft.y + shoulderright.y)/2 + (141/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx12 = shoulderright.x + (67/SHOULDERDEFINE)*shoulderlen
			righty12 = (shoulderleft.y + shoulderright.y)/2 + (126/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx13 = shoulderright.x + (75/SHOULDERDEFINE)*shoulderlen
			righty13 = (shoulderleft.y + shoulderright.y)/2 + (118/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx14 = shoulderright.x + (78/SHOULDERDEFINE)*shoulderlen
			righty14 = (shoulderleft.y + shoulderright.y)/2 + (101/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx15 = shoulderright.x + (78/SHOULDERDEFINE)*shoulderlen
			righty15 = (shoulderleft.y + shoulderright.y)/2 + (86/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx16 = shoulderright.x + (75/SHOULDERDEFINE)*shoulderlen
			righty16 = (shoulderleft.y + shoulderright.y)/2 + (50/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx17 = shoulderright.x + (71/SHOULDERDEFINE)*shoulderlen
			righty17 = (shoulderleft.y + shoulderright.y)/2 + (34/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			rightx18 = shoulderright.x + (74/SHOULDERDEFINE)*shoulderlen
			righty18 = (shoulderleft.y + shoulderright.y)/2 + (8/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])

			rx1 = int(image.shape[1]*rightx1)
			rx2 = int(image.shape[1]*rightx2)
			rx3 = int(image.shape[1]*rightx3)
			rx4 = int(image.shape[1]*rightx4)
			rx5 = int(image.shape[1]*rightx5)
			rx6 = int(image.shape[1]*rightx6)
			rx7 = int(image.shape[1]*rightx7)
			rx8 = int(image.shape[1]*rightx8)
			rx9 = int(image.shape[1]*rightx9)
			rx10 = int(image.shape[1]*rightx10)
			rx11 = int(image.shape[1]*rightx11)
			rx12 = int(image.shape[1]*rightx12)
			rx13 = int(image.shape[1]*rightx13)
			rx14 = int(image.shape[1]*rightx14)
			rx15 = int(image.shape[1]*rightx15)
			rx16 = int(image.shape[1]*rightx16)
			rx17 = int(image.shape[1]*rightx17)
			rx18 = int(image.shape[1]*rightx18)
			ry1 = int(image.shape[0]*righty1)
			ry2 = int(image.shape[0]*righty2)
			ry3 = int(image.shape[0]*righty3)
			ry4 = int(image.shape[0]*righty4)
			ry5 = int(image.shape[0]*righty5)
			ry6 = int(image.shape[0]*righty6)
			ry7 = int(image.shape[0]*righty7)
			ry8 = int(image.shape[0]*righty8)
			ry9 = int(image.shape[0]*righty9)
			ry10 = int(image.shape[0]*righty10)
			ry11 = int(image.shape[0]*righty11)
			ry12 = int(image.shape[0]*righty12)
			ry13 = int(image.shape[0]*righty13)
			ry14 = int(image.shape[0]*righty14)
			ry15 = int(image.shape[0]*righty15)
			ry16 = int(image.shape[0]*righty16)
			ry17 = int(image.shape[0]*righty17)
			ry18 = int(image.shape[0]*righty18)
			pts = np.array([[rx1,ry1],[rx2,ry2],[rx3,ry3],[rx4,ry4], [rx5,ry5],[rx6,ry6],[rx7,ry7],[rx8,ry8],[rx9,ry9],[rx10,ry10],[rx11,ry11],[rx12,ry12],[rx13,ry13],[rx14,ry14],[rx15,ry15],[rx16,ry16],[rx17,ry17],[rx18,ry18]], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(image,[pts],True,(0,0,255))

			lungx1 = shoulderright.x + (104/SHOULDERDEFINE)*shoulderlen
			lungy1 = (shoulderleft.y + shoulderright.y)/2 + (9/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx2 = shoulderright.x + (99/SHOULDERDEFINE)*shoulderlen
			lungy2 = (shoulderleft.y + shoulderright.y)/2 + (34/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx3 = shoulderright.x + (92/SHOULDERDEFINE)*shoulderlen
			lungy3 = (shoulderleft.y + shoulderright.y)/2 + (50/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx4 = shoulderright.x + (92/SHOULDERDEFINE)*shoulderlen
			lungy4 = (shoulderleft.y + shoulderright.y)/2 + (70/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx5 = shoulderright.x + (95/SHOULDERDEFINE)*shoulderlen
			lungy5 = (shoulderleft.y + shoulderright.y)/2 + (85/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx6 = shoulderright.x + (105/SHOULDERDEFINE)*shoulderlen
			lungy6 = (shoulderleft.y + shoulderright.y)/2 + (97/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx7 = shoulderright.x + (126/SHOULDERDEFINE)*shoulderlen
			lungy7 = (shoulderleft.y + shoulderright.y)/2 + (108/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx8 = shoulderright.x + (129/SHOULDERDEFINE)*shoulderlen
			lungy8 = (shoulderleft.y + shoulderright.y)/2 + (117/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx9 = shoulderright.x + (123/SHOULDERDEFINE)*shoulderlen
			lungy9 = (shoulderleft.y + shoulderright.y)/2 + (134/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx10 = shoulderright.x + (123/SHOULDERDEFINE)*shoulderlen
			lungy10 = (shoulderleft.y + shoulderright.y)/2 + (142/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx11 = shoulderright.x + (140/SHOULDERDEFINE)*shoulderlen
			lungy11 = (shoulderleft.y + shoulderright.y)/2 + (152/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx12 = shoulderright.x + (152/SHOULDERDEFINE)*shoulderlen
			lungy12 = (shoulderleft.y + shoulderright.y)/2 + (154/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx13 = shoulderright.x + (153/SHOULDERDEFINE)*shoulderlen
			lungy13 = (shoulderleft.y + shoulderright.y)/2 + (107/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx14 = shoulderright.x + (147/SHOULDERDEFINE)*shoulderlen
			lungy14 = (shoulderleft.y + shoulderright.y)/2 + (79/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx15 = shoulderright.x + (142/SHOULDERDEFINE)*shoulderlen
			lungy15 = (shoulderleft.y + shoulderright.y)/2 + (56/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx16 = shoulderright.x + (130/SHOULDERDEFINE)*shoulderlen
			lungy16 = (shoulderleft.y + shoulderright.y)/2 + (33/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx17 = shoulderright.x + (117/SHOULDERDEFINE)*shoulderlen
			lungy17 = (shoulderleft.y + shoulderright.y)/2 + (18/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lungx18 = shoulderright.x + (109/SHOULDERDEFINE)*shoulderlen
			lungy18 = (shoulderleft.y + shoulderright.y)/2 + (11/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			lx1 = int(image.shape[1]*lungx1)
			lx2 = int(image.shape[1]*lungx2)
			lx3 = int(image.shape[1]*lungx3)
			lx4 = int(image.shape[1]*lungx4)
			lx5 = int(image.shape[1]*lungx5)
			lx6 = int(image.shape[1]*lungx6)
			lx7 = int(image.shape[1]*lungx7)
			lx8 = int(image.shape[1]*lungx8)
			lx9 = int(image.shape[1]*lungx9)
			lx10 = int(image.shape[1]*lungx10)
			lx11 = int(image.shape[1]*lungx11)
			lx12 = int(image.shape[1]*lungx12)
			lx13 = int(image.shape[1]*lungx13)
			lx14 = int(image.shape[1]*lungx14)
			lx15 = int(image.shape[1]*lungx15)
			lx16 = int(image.shape[1]*lungx16)
			lx17 = int(image.shape[1]*lungx17)
			lx18 = int(image.shape[1]*lungx18)
			ly1 = int(image.shape[0]*lungy1)
			ly2 = int(image.shape[0]*lungy2)
			ly3 = int(image.shape[0]*lungy3)
			ly4 = int(image.shape[0]*lungy4)
			ly5 = int(image.shape[0]*lungy5)
			ly6 = int(image.shape[0]*lungy6)
			ly7 = int(image.shape[0]*lungy7)
			ly8 = int(image.shape[0]*lungy8)
			ly9 = int(image.shape[0]*lungy9)
			ly10 = int(image.shape[0]*lungy10)
			ly11 = int(image.shape[0]*lungy11)
			ly12 = int(image.shape[0]*lungy12)
			ly13 = int(image.shape[0]*lungy13)
			ly14 = int(image.shape[0]*lungy14)
			ly15 = int(image.shape[0]*lungy15)
			ly16 = int(image.shape[0]*lungy16)
			ly17 = int(image.shape[0]*lungy17)
			ly18 = int(image.shape[0]*lungy18)



			pts = np.array([[lx1,ly1],[lx2,ly2],[lx3,ly3],[lx4,ly4], [lx5,ly5],[lx6,ly6],[lx7,ly7],[lx8,ly8],[lx9,ly9],[lx10,ly10],[lx11,ly11],[lx12,ly12],[lx13,ly13],[lx14,ly14],[lx15,ly15],[lx16,ly16],[lx17,ly17],[lx18,ly18]], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(image,[pts],True,(0,0,255))

			

			#radius = int((30/1230)*shoulderlen*image.shape[1])
			#heartp1x = shoulderright.x + (522/1230)*shoulderlen
			#heartp1y = (shoulderleft.y + shoulderright.y)/2 + (298/1230)*shoulderlen*(image.shape[1]/image.shape[0])
			#px1 = int(image.shape[1]*heartp1x)
			#py1 = int(image.shape[0]*heartp1y)
			#cv2.circle(image,(px1,py1), radius, (0,255,0), -1)
			#heartp2x = shoulderright.x + (716/1230)*shoulderlen
			#heartp2y = (shoulderleft.y + shoulderright.y)/2 + (300/1230)*shoulderlen*(image.shape[1]/image.shape[0])
			#px2 = int(image.shape[1]*heartp2x)
			#py2 = int(image.shape[0]*heartp2y)
			#cv2.circle(image,(px2,py2), radius, (0,255,0), -1)
			#heartp3x = shoulderright.x + (717/1230)*shoulderlen
			#heartp3y = (shoulderleft.y + shoulderright.y)/2 + (632/1230)*shoulderlen*(image.shape[1]/image.shape[0])
			#px3 = int(image.shape[1]*heartp3x)
			#py3 = int(image.shape[0]*heartp3y)
			#cv2.circle(image,(px3,py3), radius, (0,255,0), -1)
			#heartp4x = shoulderright.x + (908/1230)*shoulderlen
			#heartp4y = (shoulderleft.y + shoulderright.y)/2 + (702/1230)*shoulderlen*(image.shape[1]/image.shape[0])
			#px4 = int(image.shape[1]*heartp4x)
			#py4 = int(image.shape[0]*heartp4y)
			#cv2.circle(image,(px4,py4), radius, (0,255,0), -1)
			#sx1 = int(image.shape[1]*heartx) -int(0.2*shoulderlen*image.shape[1])
			#sy1 = int(image.shape[0]*hearty) -int(0.3*shoulderlen*image.shape[1])
			#sx2 = int(image.shape[1]*heartx) 
			#sy2 = int(image.shape[0]*hearty)
			#cv2.rectangle(image,(sx1, sy1),(sx2, sy2),(0,255,0),3)
		if results.left_hand_landmarks:
			lefthandused = results.left_hand_landmarks.landmark[5]
			lefthandused.x = (results.left_hand_landmarks.landmark[6].x + results.left_hand_landmarks.landmark[7].x+ results.left_hand_landmarks.landmark[8].x)/3
			lefthandused.y = (results.left_hand_landmarks.landmark[6].y + results.left_hand_landmarks.landmark[7].y+ results.left_hand_landmarks.landmark[8].y)/3
			lefthandused.z = (results.left_hand_landmarks.landmark[6].z + results.left_hand_landmarks.landmark[7].z+ results.left_hand_landmarks.landmark[8].z)/3
			lefthandx = lefthandused.x
			lefthandy = lefthandused.y
			lx1 = int(image.shape[1]*lefthandx) - int(shoulderlen*image.shape[1]/16)
			ly1 = int(image.shape[0]*lefthandy) - int(shoulderlen*image.shape[1]/16)
			lx2 = int(image.shape[1]*lefthandx) + int(shoulderlen*image.shape[1]/16)
			ly2 = int(image.shape[0]*lefthandy) + int(shoulderlen*image.shape[1]/16)
			lx = int(image.shape[1]*lefthandx)
			ly = int(image.shape[0]*lefthandy)
			point = Point(lx, ly)
			polygon = Polygon([(rx1, ry1), (rx2, ry2), (rx3, ry3), (rx4, ry4), (rx5, ry5),(rx6,ry6),(rx7,ry7),(rx8,ry8),(rx9,ry9),(rx10,ry10),(rx11,ry11),(rx12,ry12),(rx13,ry13),(rx14,ry14), (rx15,ry15), (rx16,ry16),(rx17,ry17),(rx18,ry18)])
			polygon2 = Polygon([(lx1,ly1),(lx2,ly2),(lx3,ly3),(lx4,ly4), (lx5,ly5),(lx6,ly6),(lx7,ly7),(lx8,ly8),(lx9,ly9),(lx10,ly10),(lx11,ly11),(lx12,ly12),(lx13,ly13),(lx14,ly14),(lx15,ly15),(lx16,ly16),(lx17,ly17),(lx18,ly18)])
			if polygon.contains(point) or polygon2.contains(point):
				cv2.rectangle(image,(lx1, ly1), (lx2, ly2), (0,255,0),3)
				hearttest = True
			else:
				cv2.rectangle(image,(lx1, ly1),(lx2, ly2),(0, 170, 255),3)
			#if math.sqrt((lx-px1)**2 + (ly-py1)**2) <= ((lx2-lx1)/2-radius):
			#	aortic = True
			#if math.sqrt((lx-px2)**2 + (ly-py2)**2) <= ((lx2-lx1)/2-radius):
			#	pulmonary = True
			#if math.sqrt((lx-px3)**2 + (ly-py3)**2) <= ((lx2-lx1)/2-radius):
			#	tricuspid = True
			#if math.sqrt((lx-px4)**2 + (ly-py4)**2) <= ((lx2-lx1)/2-radius):
			#	mitral = True

		if results.right_hand_landmarks:
			righthandused = results.right_hand_landmarks.landmark[5]
			righthandused.x = (results.right_hand_landmarks.landmark[6].x + results.right_hand_landmarks.landmark[7].x+ results.right_hand_landmarks.landmark[8].x)/3
			righthandused.y = (results.right_hand_landmarks.landmark[6].y + results.right_hand_landmarks.landmark[7].y+ results.right_hand_landmarks.landmark[8].y)/3
			righthandused.z = (results.right_hand_landmarks.landmark[6].z + results.right_hand_landmarks.landmark[7].z+ results.right_hand_landmarks.landmark[8].z)/3
			righthandx = righthandused.x
			righthandy = righthandused.y
			x1 = int(image.shape[1]*righthandx) - int(shoulderlen*image.shape[1]/16)
			y1 = int(image.shape[0]*righthandy) - int(shoulderlen*image.shape[1]/16)
			x2 = int(image.shape[1]*righthandx) + int(shoulderlen*image.shape[1]/16)
			y2 = int(image.shape[0]*righthandy) + int(shoulderlen*image.shape[1]/16)
			x = int(image.shape[1]*righthandx)
			y = int(image.shape[0]*righthandy)
			point = Point(x, y)
			polygon = Polygon([(rx1, ry1), (rx2, ry2), (rx3, ry3), (rx4, ry4), (rx5, ry5),(rx6,ry6),(rx7,ry7),(rx8,ry8),(rx9,ry9),(rx10,ry10),(rx11,ry11),(rx12,ry12),(rx13,ry13),(rx14,ry14), (rx15,ry15), (rx16,ry16),(rx17,ry17),(rx18,ry18)])
			polygon2 = Polygon([(lx1,ly1),(lx2,ly2),(lx3,ly3),(lx4,ly4), (lx5,ly5),(lx6,ly6),(lx7,ly7),(lx8,ly8),(lx9,ly9),(lx10,ly10),(lx11,ly11),(lx12,ly12),(lx13,ly13),(lx14,ly14),(lx15,ly15),(lx16,ly16),(lx17,ly17),(lx18,ly18)])
			if polygon.contains(point) or polygon2.contains(point):
				cv2.rectangle(image,(x1, y1), (x2, y2), (0,255,0),3)
				hearttest = True
			else:
				cv2.rectangle(image,(x1, y1),(x2, y2),(0, 170, 255),3)
			#if math.sqrt((x-px1)**2 + (y-py1)**2) <= ((x2-x1)/2-radius):
			#	aortic = True
			#if math.sqrt((x-px2)**2 + (y-py2)**2) <= ((x2-x1)/2-radius):
			#	pulmonary = True
			#if math.sqrt((x-px3)**2 + (y-py3)**2) <= ((x2-x1)/2-radius):
			#	tricuspid = True
			#if math.sqrt((x-px4)**2 + (y-py4)**2) <= ((x2-x1)/2-radius):
			#	mitral = True
		if hearttest:
			cv2.putText(image,'Lung:',(0,75), font, 3.5,(0,255,0),2,cv2.LINE_AA)
		else:
			cv2.putText(image,'Lung:',(0,75), font, 3.5,(0,0,255),2,cv2.LINE_AA)
		#if aortic:
		#	cv2.putText(image,'Aortic',(0,125), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		#else:
		#	cv2.putText(image,'Aortic',(0,125), font, 1.5,(0,0,255),2,cv2.LINE_AA)
		#if pulmonary:
		#	cv2.putText(image,'Pulmonary',(0,175), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		#else:
		#	cv2.putText(image,'Pulmonary',(0,175), font, 1.5,(0,0,255),2,cv2.LINE_AA)
		#if tricuspid:
		#	cv2.putText(image,'Tricuspid',(0,225), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		#else:
		#	cv2.putText(image,'Tricuspid',(0,225), font, 1.5,(0,0,255),2,cv2.LINE_AA)
		#if mitral:
		#	cv2.putText(image,'Mitral',(0,275), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		#else:
		#	cv2.putText(image,'Mitral',(0,275), font, 1.5,(0,0,255),2,cv2.LINE_AA)
			

		
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.face_landmarks, mediapipe.solutions.holistic.FACEMESH_CONTOURS)
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.right_hand_landmarks, mediapipe.solutions.holistic.HAND_CONNECTIONS)
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.left_hand_landmarks, mediapipe.solutions.holistic.HAND_CONNECTIONS)
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.pose_landmarks, mediapipe.solutions.holistic.POSE_CONNECTIONS)
		

		cv2.imshow('Lung Detection', image)
		if cv2.waitKey(10) & 0xFF == ord('c'):
			break
	cap.release()
	cv2.destroyAllWindows()

