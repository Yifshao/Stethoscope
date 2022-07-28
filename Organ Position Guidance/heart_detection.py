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
		SHOULDERDEFINE = 1210
		keypoints = []
		sx1 = 0
		sx2 = 0
		sx3 = 0
		sx4 = 0
		sx5 = 0
		sx6 = 0
		sx7 = 0
		sx8 = 0
		sx9 = 0
		sx10 = 0
		sx11 = 0
		sx12 = 0
		sx13 = 0
		sx14 = 0
		sx15 = 0
		sx16 = 0
		sx17 = 0
		sx18 = 0
		sx19 = 0
		sx20 = 0
		sx21 = 0
		sx22 = 0
		sx23 = 0
		sx24 = 0
		sx25 = 0
		sx26 = 0
		sx27 = 0
		sx28 = 0
		sx29 = 0
		sx30 = 0
		sx31 = 0
		sx32 = 0
		sx33 = 0
		sx34 = 0
		sx35 = 0
		sy1 = 0
		sy2 = 0
		sy3 = 0
		sy4 = 0
		sy5 = 0
		sy6 = 0
		sy7 = 0
		sy8 = 0
		sy9 = 0
		sy10 = 0
		sy11 = 0
		sy12 = 0
		sy13 = 0
		sy14 = 0
		sy15 = 0
		sy16 = 0
		sy17 = 0
		sy18 = 0
		sy19 = 0
		sy20 = 0
		sy21 = 0
		sy22 = 0
		sy23 = 0
		sy24 = 0
		sy25 = 0
		sy26 = 0
		sy27 = 0
		sy28 = 0
		sy29 = 0
		sy30 = 0
		sy31 = 0
		sy32 = 0
		sy33 = 0
		sy34 = 0
		sy35 = 0
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
			heartx1 = shoulderright.x + (512/SHOULDERDEFINE)*shoulderlen
			hearty1 = (shoulderleft.y + shoulderright.y)/2 + (184/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx2 = shoulderright.x + (494/SHOULDERDEFINE)*shoulderlen
			hearty2 = (shoulderleft.y + shoulderright.y)/2 + (350/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx3 = shoulderright.x + (472/SHOULDERDEFINE)*shoulderlen
			hearty3 = (shoulderleft.y + shoulderright.y)/2 + (452/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx4 = shoulderright.x + (477/SHOULDERDEFINE)*shoulderlen
			hearty4 = (shoulderleft.y + shoulderright.y)/2 + (551/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx5 = shoulderright.x + (477/SHOULDERDEFINE)*shoulderlen
			hearty5 = (shoulderleft.y + shoulderright.y)/2 + (551/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx6 = shoulderright.x + (498/SHOULDERDEFINE)*shoulderlen
			hearty6 = (shoulderleft.y + shoulderright.y)/2 + (598/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx7 = shoulderright.x + (556/SHOULDERDEFINE)*shoulderlen
			hearty7 = (shoulderleft.y + shoulderright.y)/2 + (636/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx8 = shoulderright.x + (608/SHOULDERDEFINE)*shoulderlen
			hearty8 = (shoulderleft.y + shoulderright.y)/2 + (664/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx9 = shoulderright.x + (770/SHOULDERDEFINE)*shoulderlen
			hearty9 = (shoulderleft.y + shoulderright.y)/2 + (696/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx10 = shoulderright.x + (828/SHOULDERDEFINE)*shoulderlen
			hearty10 = (shoulderleft.y + shoulderright.y)/2 + (708/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx11 = shoulderright.x + (911/SHOULDERDEFINE)*shoulderlen
			hearty11 = (shoulderleft.y + shoulderright.y)/2 + (711/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx12 = shoulderright.x + (932/SHOULDERDEFINE)*shoulderlen
			hearty12 = (shoulderleft.y + shoulderright.y)/2 + (656/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx13 = shoulderright.x + (927/SHOULDERDEFINE)*shoulderlen
			hearty13 = (shoulderleft.y + shoulderright.y)/2 + (587/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx14 = shoulderright.x + (922/SHOULDERDEFINE)*shoulderlen
			hearty14 = (shoulderleft.y + shoulderright.y)/2 + (541/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx15 = shoulderright.x + (913/SHOULDERDEFINE)*shoulderlen
			hearty15 = (shoulderleft.y + shoulderright.y)/2 + (517/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx16 = shoulderright.x + (885/SHOULDERDEFINE)*shoulderlen
			hearty16 = (shoulderleft.y + shoulderright.y)/2 + (432/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx17 = shoulderright.x + (837/SHOULDERDEFINE)*shoulderlen
			hearty17 = (shoulderleft.y + shoulderright.y)/2 + (361/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx18 = shoulderright.x + (793/SHOULDERDEFINE)*shoulderlen
			hearty18 = (shoulderleft.y + shoulderright.y)/2 + (327/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx19 = shoulderright.x + (755/SHOULDERDEFINE)*shoulderlen
			hearty19 = (shoulderleft.y + shoulderright.y)/2 + (314/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx20 = shoulderright.x + (797/SHOULDERDEFINE)*shoulderlen
			hearty20 = (shoulderleft.y + shoulderright.y)/2 + (308/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx21 = shoulderright.x + (806/SHOULDERDEFINE)*shoulderlen
			hearty21 = (shoulderleft.y + shoulderright.y)/2 + (264/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx22 = shoulderright.x + (790/SHOULDERDEFINE)*shoulderlen
			hearty22 = (shoulderleft.y + shoulderright.y)/2 + (236/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx23 = shoulderright.x + (760/SHOULDERDEFINE)*shoulderlen
			hearty23 = (shoulderleft.y + shoulderright.y)/2 + (233/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx24 = shoulderright.x + (750/SHOULDERDEFINE)*shoulderlen
			hearty24 = (shoulderleft.y + shoulderright.y)/2 + (184/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx25 = shoulderright.x + (731/SHOULDERDEFINE)*shoulderlen
			hearty25 = (shoulderleft.y + shoulderright.y)/2 + (147/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx26 = shoulderright.x + (703/SHOULDERDEFINE)*shoulderlen
			hearty26 = (shoulderleft.y + shoulderright.y)/2 + (134/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx27 = shoulderright.x + (700/SHOULDERDEFINE)*shoulderlen
			hearty27 = (shoulderleft.y + shoulderright.y)/2 + (78/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx28 = shoulderright.x + (674/SHOULDERDEFINE)*shoulderlen
			hearty28 = (shoulderleft.y + shoulderright.y)/2 + (79/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx29 = shoulderright.x + (637/SHOULDERDEFINE)*shoulderlen
			hearty29 = (shoulderleft.y + shoulderright.y)/2 + (68/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx30 = shoulderright.x + (633/SHOULDERDEFINE)*shoulderlen
			hearty30 = (shoulderleft.y + shoulderright.y)/2 + (99/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx31 = shoulderright.x + (591/SHOULDERDEFINE)*shoulderlen
			hearty31 = (shoulderleft.y + shoulderright.y)/2 + (96/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx32 = shoulderright.x + (605/SHOULDERDEFINE)*shoulderlen
			hearty32 = (shoulderleft.y + shoulderright.y)/2 + (147/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx33 = shoulderright.x + (583/SHOULDERDEFINE)*shoulderlen
			hearty33 = (shoulderleft.y + shoulderright.y)/2 + (185/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx34 = shoulderright.x + (555/SHOULDERDEFINE)*shoulderlen
			hearty34 = (shoulderleft.y + shoulderright.y)/2 + (161/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			heartx35 = shoulderright.x + (527/SHOULDERDEFINE)*shoulderlen
			hearty35 = (shoulderleft.y + shoulderright.y)/2 + (175/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])

			sx1 = int(image.shape[1]*heartx1)
			sx2 = int(image.shape[1]*heartx2)
			sx3 = int(image.shape[1]*heartx3)
			sx4 = int(image.shape[1]*heartx4)
			sx5 = int(image.shape[1]*heartx5)
			sx6 = int(image.shape[1]*heartx6)
			sx7 = int(image.shape[1]*heartx7)
			sx8 = int(image.shape[1]*heartx8)
			sx9 = int(image.shape[1]*heartx9)
			sx10 = int(image.shape[1]*heartx10)
			sx11 = int(image.shape[1]*heartx11)
			sx12 = int(image.shape[1]*heartx12)
			sx13 = int(image.shape[1]*heartx13)
			sx14 = int(image.shape[1]*heartx14)
			sx15 = int(image.shape[1]*heartx15)
			sx16 = int(image.shape[1]*heartx16)
			sx17 = int(image.shape[1]*heartx17)
			sx18 = int(image.shape[1]*heartx18)
			sx19 = int(image.shape[1]*heartx19)
			sx20 = int(image.shape[1]*heartx20)
			sx21 = int(image.shape[1]*heartx21)
			sx22 = int(image.shape[1]*heartx22)
			sx23 = int(image.shape[1]*heartx23)
			sx24 = int(image.shape[1]*heartx24)
			sx25 = int(image.shape[1]*heartx25)
			sx26 = int(image.shape[1]*heartx26)
			sx27 = int(image.shape[1]*heartx27)
			sx28 = int(image.shape[1]*heartx28)
			sx29 = int(image.shape[1]*heartx29)
			sx30 = int(image.shape[1]*heartx30)
			sx31 = int(image.shape[1]*heartx31)
			sx32 = int(image.shape[1]*heartx32)
			sx33 = int(image.shape[1]*heartx33)
			sx34 = int(image.shape[1]*heartx34)
			sx35 = int(image.shape[1]*heartx35)

			sy1 = int(image.shape[0]*hearty1)
			sy2 = int(image.shape[0]*hearty2)
			sy3 = int(image.shape[0]*hearty3)
			sy4 = int(image.shape[0]*hearty4)
			sy5 = int(image.shape[0]*hearty5)
			sy6 = int(image.shape[0]*hearty6)
			sy7 = int(image.shape[0]*hearty7)
			sy8 = int(image.shape[0]*hearty8)
			sy9 = int(image.shape[0]*hearty9)
			sy10 = int(image.shape[0]*hearty10)
			sy11 = int(image.shape[0]*hearty11)
			sy12 = int(image.shape[0]*hearty12)
			sy13 = int(image.shape[0]*hearty13)
			sy14 = int(image.shape[0]*hearty14)
			sy15 = int(image.shape[0]*hearty15)
			sy16 = int(image.shape[0]*hearty16)
			sy17 = int(image.shape[0]*hearty17)
			sy18 = int(image.shape[0]*hearty18)
			sy19 = int(image.shape[0]*hearty19)
			sy20 = int(image.shape[0]*hearty20)
			sy21 = int(image.shape[0]*hearty21)
			sy22 = int(image.shape[0]*hearty22)
			sy23 = int(image.shape[0]*hearty23)
			sy24 = int(image.shape[0]*hearty24)
			sy25 = int(image.shape[0]*hearty25)
			sy26 = int(image.shape[0]*hearty26)
			sy27 = int(image.shape[0]*hearty27)
			sy28 = int(image.shape[0]*hearty28)
			sy29 = int(image.shape[0]*hearty29)
			sy30 = int(image.shape[0]*hearty30)
			sy31 = int(image.shape[0]*hearty31)
			sy32 = int(image.shape[0]*hearty32)
			sy33 = int(image.shape[0]*hearty33)
			sy34 = int(image.shape[0]*hearty34)
			sy35 = int(image.shape[0]*hearty35)
			pts = np.array([[sx1,sy1],[sx2,sy2],[sx3,sy3],[sx4,sy4],[sx5,sy5],[sx6,sy6],[sx7,sy7],[sx8,sy8],[sx9,sy9],[sx10,sy10],[sx11,sy11],[sx12,sy12],[sx13,sy13],[sx14,sy14],[sx15,sy15],[sx16,sy16],[sx17,sy17],[sx18,sy18],[sx19,sy19],[sx20,sy20],[sx21,sy21],[sx22,sy22],[sx23,sy23],[sx24,sy24],[sx25,sy25],[sx26,sy26],[sx27,sy27],[sx28,sy28],[sx29,sy29],[sx30,sy30],[sx31,sy31],[sx32,sy32],[sx33,sy33],[sx34,sy34],[sx35,sy35]], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(image,[pts],True,(0,0,255))
			radius = int((30/SHOULDERDEFINE)*shoulderlen*image.shape[1])
			heartp1x = shoulderright.x + (523/SHOULDERDEFINE)*shoulderlen
			heartp1y = (shoulderleft.y + shoulderright.y)/2 + (292/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			px1 = int(image.shape[1]*heartp1x)
			py1 = int(image.shape[0]*heartp1y)
			cv2.circle(image,(px1,py1), radius, (0,255,0), -1)
			heartp2x = shoulderright.x + (717/SHOULDERDEFINE)*shoulderlen
			heartp2y = (shoulderleft.y + shoulderright.y)/2 + (298/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			px2 = int(image.shape[1]*heartp2x)
			py2 = int(image.shape[0]*heartp2y)
			cv2.circle(image,(px2,py2), radius, (0,255,0), -1)
			heartp3x = shoulderright.x + (719/SHOULDERDEFINE)*shoulderlen
			heartp3y = (shoulderleft.y + shoulderright.y)/2 + (631/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			px3 = int(image.shape[1]*heartp3x)
			py3 = int(image.shape[0]*heartp3y)
			cv2.circle(image,(px3,py3), radius, (0,255,0), -1)
			heartp4x = shoulderright.x + (912/SHOULDERDEFINE)*shoulderlen
			heartp4y = (shoulderleft.y + shoulderright.y)/2 + (698/SHOULDERDEFINE)*shoulderlen*(image.shape[1]/image.shape[0])
			px4 = int(image.shape[1]*heartp4x)
			py4 = int(image.shape[0]*heartp4y)
			cv2.circle(image,(px4,py4), radius, (0,255,0), -1)
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
			polygon = Polygon([(sx1, sy1), (sx2, sy2), (sx3, sy3), (sx4, sy4), (sx5, sy5),(sx6,sy6),(sx7,sy7),(sx8,sy8),(sx9,sy9),(sx10,sy10),(sx11,sy11),(sx12,sy12),(sx13,sy13),(sx14,sy14), (sx15,sy15), (sx16,sy16),(sx17,sy17),(sx18,sy18),(sx19,sy19),(sx20,sy20),(sx21,sy21),(sx22,sy22),(sx23,sy23),(sx24,sy24),(sx25,sy25),(sx26,sy26),(sx27,sy27),(sx28,sy28),(sx29,sy29),(sx30,sy30),(sx31,sy31),(sx32,sy32),(sx33,sy33),(sx34,sy34),(sx35,sy35)])
			if polygon.contains(point):
				cv2.rectangle(image,(lx1, ly1), (lx2, ly2), (0,255,0),3)
				hearttest = True
			else:
				cv2.rectangle(image,(lx1, ly1),(lx2, ly2),(0, 170, 255),3)
			if math.sqrt((lx-px1)**2 + (ly-py1)**2) <= ((lx2-lx1)/2-radius):
				aortic = True
			if math.sqrt((lx-px2)**2 + (ly-py2)**2) <= ((lx2-lx1)/2-radius):
				pulmonary = True
			if math.sqrt((lx-px3)**2 + (ly-py3)**2) <= ((lx2-lx1)/2-radius):
				tricuspid = True
			if math.sqrt((lx-px4)**2 + (ly-py4)**2) <= ((lx2-lx1)/2-radius):
				mitral = True

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
			polygon = Polygon([(sx1, sy1), (sx2, sy2), (sx3, sy3), (sx4, sy4), (sx5, sy5),(sx6,sy6),(sx7,sy7),(sx8,sy8),(sx9,sy9),(sx10,sy10),(sx11,sy11),(sx12,sy12),(sx13,sy13),(sx14,sy14), (sx15,sy15), (sx16,sy16),(sx17,sy17),(sx18,sy18),(sx19,sy19),(sx20,sy20),(sx21,sy21),(sx22,sy22),(sx23,sy23),(sx24,sy24),(sx25,sy25),(sx26,sy26),(sx27,sy27),(sx28,sy28),(sx29,sy29),(sx30,sy30),(sx31,sy31),(sx32,sy32),(sx33,sy33),(sx34,sy34),(sx35,sy35)])
			if polygon.contains(point):
				cv2.rectangle(image,(x1, y1), (x2, y2), (0,255,0),3)
				hearttest = True
			else:
				cv2.rectangle(image,(x1, y1),(x2, y2),(0, 170, 255),3)
			if math.sqrt((x-px1)**2 + (y-py1)**2) <= ((x2-x1)/2-radius):
				aortic = True
			if math.sqrt((x-px2)**2 + (y-py2)**2) <= ((x2-x1)/2-radius):
				pulmonary = True
			if math.sqrt((x-px3)**2 + (y-py3)**2) <= ((x2-x1)/2-radius):
				tricuspid = True
			if math.sqrt((x-px4)**2 + (y-py4)**2) <= ((x2-x1)/2-radius):
				mitral = True
		if hearttest:
			cv2.putText(image,'Heart:',(0,75), font, 3.5,(0,255,0),2,cv2.LINE_AA)
		else:
			cv2.putText(image,'Heart:',(0,75), font, 3.5,(0,0,255),2,cv2.LINE_AA)
		if aortic:
			cv2.putText(image,'Aortic',(0,125), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		else:
			cv2.putText(image,'Aortic',(0,125), font, 1.5,(0,0,255),2,cv2.LINE_AA)
		if pulmonary:
			cv2.putText(image,'Pulmonary',(0,175), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		else:
			cv2.putText(image,'Pulmonary',(0,175), font, 1.5,(0,0,255),2,cv2.LINE_AA)
		if tricuspid:
			cv2.putText(image,'Tricuspid',(0,225), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		else:
			cv2.putText(image,'Tricuspid',(0,225), font, 1.5,(0,0,255),2,cv2.LINE_AA)
		if mitral:
			cv2.putText(image,'Mitral',(0,275), font, 1.5,(0,255,0),2,cv2.LINE_AA)
		else:
			cv2.putText(image,'Mitral',(0,275), font, 1.5,(0,0,255),2,cv2.LINE_AA)
			

		
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.face_landmarks, mediapipe.solutions.holistic.FACEMESH_CONTOURS)
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.right_hand_landmarks, mediapipe.solutions.holistic.HAND_CONNECTIONS)
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.left_hand_landmarks, mediapipe.solutions.holistic.HAND_CONNECTIONS)
		#mediapipe.solutions.drawing_utils.draw_landmarks(image,results.pose_landmarks, mediapipe.solutions.holistic.POSE_CONNECTIONS)
		

		cv2.imshow('Heart Detection', image)
		if cv2.waitKey(10) & 0xFF == ord('c'):
			break
	cap.release()
	cv2.destroyAllWindows()

