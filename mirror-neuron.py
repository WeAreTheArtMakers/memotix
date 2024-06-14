import cv2
import time

class MirrorNeuron:
    def __init__(self):
        self.observed_actions = []
        self.imitated_actions = []

    def observe_action(self, action):
        """Bir hareketi gözlemle."""
        self.observed_actions.append(action)

    def imitate_action(self):
        """Gözlemlenen hareketi taklit et."""
        if self.observed_actions:
            action_to_imitate = self.observed_actions.pop(0)
            self.imitated_actions.append(action_to_imitate)
            print(f"Taklit edilen hareket: {action_to_imitate}")
        else:
            print("Gözlemlenecek hareket yok.")

# Hareketleri algılama ve gözlemleme
def detect_movement():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılamadı.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    last_frame = blur
    movement_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if movement_detected:
            diff = cv2.absdiff(last_frame, blur)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                mirror_neuron_system.observe_action("Hareket Algılandı")

        last_frame = blur.copy()
        movement_detected = True

        cv2.imshow('Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ayna nöron sistemi örneği
mirror_neuron_system = MirrorNeuron()

# Kamera ile hareket algılama
detect_movement()

# Hareketlerin taklit edilmesi
while mirror_neuron_system.observed_actions:
    mirror_neuron_system.imitate_action()
    time.sleep(1)