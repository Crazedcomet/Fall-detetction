import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
import requests  # Required for notifications

# --- TELEGRAM CONFIG ---
# Replace these with your actual details from BotFather and userinfobot
TELEGRAM_TOKEN = "8101652572:AAE1LKHn5Q3kr0VkvkSxu7AnHWzL-fadX68"
TELEGRAM_CHAT_ID = "8687487278"

def send_telegram_notification(message):
    """Sends a message to your phone via Telegram."""
    if not TELEGRAM_TOKEN or "YOUR_BOT" in TELEGRAM_TOKEN:
        print("[SKIP] Telegram not configured correctly.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
        print(f"[SUCCESS] Notification sent: {message}")
    except Exception as e:
        print(f"[ERROR] Failed to send notification: {e}")

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (shoulders, hips, etc.)
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    esp32_cam_url = "http://10.65.70.137:81/stream"
    print(f"[INFO] Connecting to ESP32-CAM Stream at {esp32_cam_url}...")
    cap = cv2.VideoCapture(esp32_cam_url)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open stream from {esp32_cam_url}")
        print("[TIP] Make sure your phone hotspot is ON and the PC is on the same network.")
        return

    fall_detected = False
    fall_start_time = 0
    last_notification_time = 0 
    
    print("Starting Fall Detection... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # keypoints
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
            mid_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
            
            dy = abs(mid_shoulder[1] - mid_hip[1])
            dx = abs(mid_shoulder[0] - mid_hip[0])
            
            is_horizontal = dx > dy * 0.9 
            
            if is_horizontal:
                if not fall_detected:
                    fall_detected = True
                    fall_start_time = time.time()
                
                if time.time() - fall_start_time > 0.5: 
                    cv2.putText(image, "FALL DETECTED!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                    
                    threading.Thread(target=play_alarm, daemon=True).start()

                    current_time = time.time()
                    if current_time - last_notification_time > 60:
                        msg = "ALERT: A fall has been detected by the health monitor!"
                        threading.Thread(target=send_telegram_notification, args=(msg,), daemon=True).start()
                        last_notification_time = current_time
            else:
                fall_detected = False
                fall_start_time = 0
                
        except Exception:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)               
        cv2.imshow('Fall Detection Feed (ESP32-CAM)', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
