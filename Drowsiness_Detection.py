from scipy.spatial import distance
import imutils
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core as mp_core
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode
import cv2
import numpy as np
import winsound
from threading import Thread
import time
import smtplib
from email.message import EmailMessage
import os
import csv

# ==========================================
# SMS & EMAIL NOTIFICATION CONFIGURATION
# ==========================================
# To activate: Replace the strings below with your actual data.
SENDER_EMAIL = "mohansanjai1716@gmail.com"
SENDER_PASSWORD = "ynsnafpxrckfoybj"  # MUST be a Google App Password (no spaces)
RECEIVER_EMAIL = "mohansanjai1716@gmail.com" 
# Note: SMS gateways (like @vtext) are mostly for US carriers. If you are outside the US, this text may not deliver, but the Email will arrive instantly!
RECEIVER_SMS = "9514210203@vtext.com" 
# ==========================================

ALARM_ON = False
LAST_EMAIL_TIME = 0
last_screenshot_time = 0

# --- Logging Initialization ---
for folder in ["DROWSINESS", "YAWNING", "DISTRACTED"]:
	os.makedirs(f"alerts/{folder}", exist_ok=True)

csv_filename = "alerts_log.csv"
if not os.path.isfile(csv_filename):
	with open(csv_filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Timestamp", "Alert_Type", "Image_File"])
# ------------------------------

def send_email_notification(alert_type):
	global LAST_EMAIL_TIME
	
	# Safety check so it doesn't crash if you haven't entered your email
	if SENDER_EMAIL == "your_email@gmail.com" or SENDER_PASSWORD == "your_16_digit_app_password":
		print(f"\n[ALERT] {alert_type} detected! (Live transmission disabled - Please configure credentials at the top of the script)")
		return

	# Rate limit: Only send 1 email every 60 seconds
	if time.time() - LAST_EMAIL_TIME < 60:
		return
	LAST_EMAIL_TIME = time.time()
	
	try:
		print(f"\n[ALERT] Transmitting live {alert_type} notification email and SMS...")
		
		# Connect to Google's secured SMTP server
		server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
		server.login(SENDER_EMAIL, SENDER_PASSWORD)
		
		targets = [RECEIVER_EMAIL, RECEIVER_SMS]
		for target in targets:
			if "@" in target and "1234567890" not in target and "example.com" not in target:
				msg = EmailMessage()
				msg.set_content(f"URGENT: The driver monitoring system has triggered a {alert_type} alert at {time.ctime()}!")
				msg['Subject'] = f'Driver Alert: {alert_type} Detected'
				msg['From'] = SENDER_EMAIL
				msg['To'] = target
				server.send_message(msg)
				print(f" -> Sent to {target}")
				
		server.quit()
		print("[ALERT] Live Notifications successfully transmitted.")
	except Exception as e:
		print(f"[ERROR] Failed to send live notification. Check your App Password! Error: {e}")

def sleep_while_active(duration):
	start_sleep = time.time()
	while ALARM_ON and (time.time() - start_sleep) < duration:
		time.sleep(0.05)

def sound_alarm():
	global ALARM_ON
	alarm_start = time.time()
	continuous_playing = False
	
	while ALARM_ON:
		elapsed = time.time() - alarm_start
		if elapsed < 3.0:
			# Level 1: Initial alert (Lower pitch, spaced out)
			winsound.Beep(1000, 300)
			sleep_while_active(0.7)
		elif elapsed < 6.0:
			# Level 2: Rising urgency (Medium pitch, faster)
			winsound.Beep(1500, 300)
			sleep_while_active(0.3)
		else:
			# Level 3: Critical Continuous Alarm
			if not continuous_playing:
				# Uses built-in Windows critical sound, continuously looped asynchronously
				winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_LOOP | winsound.SND_ASYNC)
				continuous_playing = True
			sleep_while_active(0.5)

	# Clean up to stop the looping sound instantly when awake
	if continuous_playing:
		winsound.PlaySound(None, winsound.SND_PURGE)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
	# mouth[0]=top inner, mouth[1]=bottom inner, mouth[2]=left inner, mouth[3]=right inner
	A = distance.euclidean(mouth[0], mouth[1])
	B = distance.euclidean(mouth[2], mouth[3])
	mar = A / B
	return mar

thresh = 0.25
frame_check = 20
mar_thresh = 0.6
yawn_frame_check = 10
distract_frame_check = 20

model_path = "D:/project/lung/Drowsiness_Detection/face_landmarker.task"
base_options = mp_core.base_options.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_running_mode.VisionTaskRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

import time

cap = cv2.VideoCapture(0)
flag = 0
yawn_flag = 0
distract_flag = 0
alarm_cooldown = 0
frame_timestamp_ms = 0

while True:
	ret, frame = cap.read()
	if not ret:
		break
	frame = imutils.resize(frame, width=450)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	mp_frame = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
	frame_timestamp_ms = int(time.time() * 1000)
	result = face_mesh.detect_for_video(mp_frame, frame_timestamp_ms)
	if result.face_landmarks:
		for landmarks in result.face_landmarks:
			h, w, _ = frame.shape
			# Correct Mediapipe landmarks for EAR (0: outer, 1: top, 2: top, 3: inner, 4: bottom, 5: bottom)
			left_eye_points = [33, 160, 158, 133, 153, 144]
			right_eye_points = [362, 385, 387, 263, 373, 380]
			leftEye = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in left_eye_points], dtype=np.int32)
			rightEye = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in right_eye_points], dtype=np.int32)
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
			cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)
			
			# MAR calculation
			mouth_points = [13, 14, 78, 308]
			mouth = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in mouth_points], dtype=np.int32)
			mar = mouth_aspect_ratio(mouth)
			cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)
			
			# Head Pose Estimation via solvePnP
			face_2d = []
			for idx in [1, 33, 263, 61, 291, 152]:
				lm = landmarks[idx]
				x, y = int(lm.x * w), int(lm.y * h)
				face_2d.append([x, y])
			
			face_2d = np.array(face_2d, dtype=np.float64)
			
			# Generic 3D model points in World Coordinates
			face_3d = np.array([
				[0.0, 0.0, 0.0],            # 1: Nose tip
				[-165.0, -170.0, -135.0],   # 33: Left eye outer corner
				[165.0, -170.0, -135.0],    # 263: Right eye outer corner
				[-150.0, 150.0, -125.0],    # 61: Left Mouth corner
				[150.0, 150.0, -125.0],     # 291: Right mouth corner
				[0.0, 330.0, -65.0]         # 152: Chin
			], dtype=np.float64)
			
			focal_length = 1 * w
			camera_matrix = np.array([[focal_length, 0, w/2],
									  [0, focal_length, h/2],
									  [0, 0, 1]], dtype=np.float64)
			distortion_matrix = np.zeros((4, 1), dtype=np.float64)
			
			success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, camera_matrix, distortion_matrix)
			rmat, jac = cv2.Rodrigues(rot_vec)
			angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
			
			pitch = angles[0]
			yaw = angles[1]
			
			# Gaze Tracking (Iris vs Eye Corners)
			gaze_distracted = False
			avg_gaze = 0.5
			if len(landmarks) >= 478:
				p_left_l  = np.array([landmarks[33].x * w, landmarks[33].y * h])
				p_right_l = np.array([landmarks[133].x * w, landmarks[133].y * h])
				p_iris_l  = np.array([landmarks[468].x * w, landmarks[468].y * h])
				
				p_left_r  = np.array([landmarks[362].x * w, landmarks[362].y * h])
				p_right_r = np.array([landmarks[263].x * w, landmarks[263].y * h])
				p_iris_r  = np.array([landmarks[473].x * w, landmarks[473].y * h])

				w_l, d_l = distance.euclidean(p_left_l, p_right_l), distance.euclidean(p_left_l, p_iris_l)
				r_l = d_l / w_l if w_l > 0 else 0.5
				
				w_r, d_r = distance.euclidean(p_left_r, p_right_r), distance.euclidean(p_left_r, p_iris_r)
				r_r = d_r / w_r if w_r > 0 else 0.5
				
				avg_gaze = (r_l + r_r) / 2.0
				if avg_gaze < 0.42 or avg_gaze > 0.58:
					gaze_distracted = True

			cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			cv2.putText(frame, f"PITCH: {pitch:.1f} YAW: {yaw:.1f} GAZE: {avg_gaze:.2f}", (10, h - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
				
			drowsy = False
			yawning = False
			distracted = False

			# Distracted thresholds: extreme head yaw/pitch, or eyes looking fully away
			if yaw < -12 or yaw > 12 or pitch < -20 or gaze_distracted:
				distract_flag += 1
				if distract_flag >= distract_frame_check:
					distracted = True
			else:
				distract_flag = 0

			if ear < thresh:
				flag += 1
				if flag >= frame_check:
					drowsy = True
			else:
				flag = 0

			if mar > mar_thresh:
				yawn_flag += 1
				if yawn_flag >= yawn_frame_check:
					yawning = True
			else:
				yawn_flag = 0

			if drowsy:
				cv2.putText(frame, "****************DROWSY ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************DROWSY ALERT!****************", (10, 325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
			if yawning:
				cv2.putText(frame, "****************YAWN ALERT!****************", (10, 80),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

			if distracted:
				cv2.putText(frame, "**************DISTRACTED ALERT!**************", (10, 130),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

			if drowsy or yawning or distracted:
				alarm_cooldown = 20  # Provide ~1 second buffer to smooth out camera jitters
				
				if drowsy: alert_reason = "DROWSINESS"
				elif yawning: alert_reason = "YAWNING"
				else: alert_reason = "DISTRACTED"

				# Take a screenshot and log it every 1 second continuously during the event!
				if time.time() - last_screenshot_time >= 1.0:
					last_screenshot_time = time.time()
					timestamp_str = time.strftime("%Y%m%d-%H%M%S")
					# Store in nicely organized separate subfolders!
					img_filename = f"alerts/{alert_reason}/{alert_reason}_{timestamp_str}.jpg"
					cv2.imwrite(img_filename, frame)
					
					# Append to CSV log safely
					try:
						with open(csv_filename, mode='a', newline='') as file:
							writer = csv.writer(file)
							writer.writerow([time.ctime(), alert_reason, img_filename])
					except PermissionError:
						print(f"[WARNING] Could not update {csv_filename} (File is currently locked or open).")

				if not ALARM_ON:
					# Fire email/SMS safely asynchronously (already rate-limited to 60s natively)
					Thread(target=send_email_notification, args=(alert_reason,), daemon=True).start()
					
					# Start the escalating audio siren thread
					ALARM_ON = True
					Thread(target=sound_alarm, daemon=True).start()
			else:
				alarm_cooldown -= 1
				if alarm_cooldown <= 0:
					ALARM_ON = False
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

face_mesh.close()
cv2.destroyAllWindows()
cap.release() 
