import cv2
import mediapipe as mp
import dlib
from fer import FER
import pyautogui

# MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# dlib face detection and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# FER for emotion detection
emotion_detector = FER()

# Duygu haritası
emotion_map = {
    "angry": "Sinirli",
    "disgust": "Tiksinti",
    "fear": "Korku",
    "happy": "Mutlu",
    "sad": "Uzgun",
    "surprise": "Saskin",
    "neutral": "Notr"
}

class MirrorNeuron:
    def __init__(self):
        self.observed_actions = []
        self.imitated_actions = []
        self.current_emotions = {}
        self.mouse_control_enabled = False

    def observe_action(self, action, location, emotions=None):
        """Observe an action."""
        self.observed_actions.append((action, location, emotions))
        if emotions:
            self.current_emotions = emotions
            self.mouse_control_enabled = emotions.get("Mutlu", 0) > 0.5 and emotions.get("Üzgün", 0) < 0.5

    def imitate_action(self):
        """Imitate an observed action."""
        if self.observed_actions:
            action_to_imitate, location, emotions = self.observed_actions.pop(0)
            self.imitated_actions.append(action_to_imitate)
            print(f"Imitated action: {action_to_imitate}, Location: {location}, Emotions: {emotions}")
        else:
            print("No action to imitate.")

def draw_geometric_face_parts(frame, landmarks):
    points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
    
    # Jaw line
    for i in range(1, 17):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    
    # Eyebrow lines
    for i in range(18, 22):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    for i in range(23, 27):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    
    # Nose line
    for i in range(28, 31):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    for i in range(32, 36):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    
    # Eye lines
    for i in range(37, 42):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    cv2.line(frame, points[36], points[41], (0, 255, 0), 2)
    for i in range(43, 48):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    cv2.line(frame, points[42], points[47], (0, 255, 0), 2)
    
    # Mouth lines
    for i in range(49, 60):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    cv2.line(frame, points[48], points[59], (0, 255, 0), 2)
    for i in range(61, 68):
        cv2.line(frame, points[i], points[i-1], (0, 255, 0), 2)
    cv2.line(frame, points[60], points[67], (0, 255, 0), 2)

def detect_movement(mirror_neuron_system):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))  # Reduce resolution
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Draw circle at fingertips
                    if id in [4, 8, 12, 16, 20]:
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                        mirror_neuron_system.observe_action("Fingertip Detected", (cx, cy))
                        
                        if mirror_neuron_system.mouse_control_enabled:
                            pyautogui.moveTo(cx, cy)

        # Face detection and geometric drawing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            draw_geometric_face_parts(frame, landmarks)
            
            # Emotion detection
            emotions = emotion_detector.detect_emotions(frame[face.top():face.bottom(), face.left():face.right()])
            if emotions:
                translated_emotions = {emotion_map[k]: v for k, v in emotions[0]['emotions'].items()}
                mirror_neuron_system.observe_action("Face Detected", (face.left(), face.top(), face.width(), face.height()), translated_emotions)
            
        # Display emotions in the top left corner
        y_offset = 20
        for emotion, score in mirror_neuron_system.current_emotions.items():
            cv2.putText(frame, f"{emotion}: {score:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30

        cv2.imshow('Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

mirror_neuron_system = MirrorNeuron()
detect_movement(mirror_neuron_system)

while mirror_neuron_system.observed_actions:
    mirror_neuron_system.imitate_action()
    time.sleep(1)