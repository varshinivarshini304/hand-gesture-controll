import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from enum import Enum

# ─────────────────────────────────────────────
# Safety Settings
# ─────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# ─────────────────────────────────────────────
# Performance & Gesture Tuning
# ─────────────────────────────────────────────
FRAME_RATE             = 30
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING_FACTOR       = 0.5
CLICK_HOLD_THRESHOLD   = 0.3   # Seconds to hold for click
DRAG_HOLD_THRESHOLD    = 0.6   # Seconds to hold for drag (longer than click)
SCROLL_COOLDOWN        = 0.15
VOLUME_COOLDOWN        = 0.1
ZOOM_COOLDOWN          = 0.2
ACTION_COOLDOWN        = 0.4   # General action cooldown

# ─────────────────────────────────────────────
# Gesture States
# ─────────────────────────────────────────────
class GestureState(Enum):
    NONE         = "NONE"
    MOVE         = "MOVE"
    LEFT_CLICK   = "LEFT_CLICK"
    RIGHT_CLICK  = "RIGHT_CLICK"
    DRAG         = "DRAG"
    DRAGGING     = "DRAGGING"
    SCROLL_UP    = "SCROLL_UP"
    SCROLL_DOWN  = "SCROLL_DOWN"
    VOLUME_UP    = "VOLUME_UP"
    VOLUME_DOWN  = "VOLUME_DOWN"
    ZOOM_IN      = "ZOOM_IN"
    ZOOM_OUT     = "ZOOM_OUT"
    MINIMIZE     = "MINIMIZE"

# ─────────────────────────────────────────────
# Gesture Controller
# ─────────────────────────────────────────────
class GestureController:
    def __init__(self):
        self.last_action        = GestureState.NONE
        self.gesture_start_time = 0
        self.is_dragging        = False
        self.last_scroll_time   = 0
        self.last_volume_time   = 0
        self.last_zoom_time     = 0
        self.last_action_time   = 0
        self.smoothed_pos       = None

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.6
        )
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

    # ─── Finger Counting ─────────────────────
    def count_fingers(self, landmarks, hand_label):
        """
        Returns list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
        True = finger is extended/up
        """
        fingers = []

        # Thumb: compare x-axis based on hand orientation
        if hand_label == 'Right':
            fingers.append(landmarks[4].x < landmarks[3].x)
        else:
            fingers.append(landmarks[4].x > landmarks[3].x)

        # Four fingers: tip.y < pip.y means finger is up
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            fingers.append(landmarks[tip].y < landmarks[pip].y)

        return fingers  # [Thumb, Index, Middle, Ring, Pinky]

    # ─── Gesture Classification ───────────────
    def classify_gesture(self, fingers):
        """
        Map finger states to GestureState.
        fingers = [Thumb, Index, Middle, Ring, Pinky]
        """
        T, I, M, R, P = fingers
        total = sum(fingers)

        # ── Closed fist ──────────────────────
        if total == 0:
            return GestureState.MINIMIZE

        # ── Single fingers ───────────────────
        if T and not I and not M and not R and not P:
            return GestureState.VOLUME_UP

        if P and not T and not I and not M and not R:
            return GestureState.VOLUME_DOWN

        # ── Pinch gestures (Thumb + Index) ───
        # Used for click AND drag (duration determines which)
        if T and I and not M and not R and not P:
            return GestureState.LEFT_CLICK  # execute_action promotes to DRAG

        # ── Right Click (Thumb + Middle) ─────
        if T and M and not I and not R and not P:
            return GestureState.RIGHT_CLICK

        # ── Scroll (Index + Middle only) ─────
        if I and M and not R and not P:
            # Thumb UP = scroll DOWN (push down motion)
            # Thumb DOWN = scroll UP (pull up motion)
            return GestureState.SCROLL_DOWN if T else GestureState.SCROLL_UP

        # ── Zoom (Index + Middle + Ring) ─────
        if I and M and R and not P:
            return GestureState.ZOOM_IN if T else GestureState.ZOOM_OUT

        # ── Default: move cursor ─────────────
        return GestureState.MOVE

    # ─── Mouse Smoothing ──────────────────────
    def get_smoothed_pos(self, raw_pos):
        if self.smoothed_pos is None:
            self.smoothed_pos = raw_pos
        else:
            sx = self.smoothed_pos[0] * SMOOTHING_FACTOR + raw_pos[0] * (1 - SMOOTHING_FACTOR)
            sy = self.smoothed_pos[1] * SMOOTHING_FACTOR + raw_pos[1] * (1 - SMOOTHING_FACTOR)
            self.smoothed_pos = (sx, sy)
        return self.smoothed_pos

    # ─── Action Executor ──────────────────────
    def execute_action(self, gesture, landmarks):
        now = time.time()

        # ── Always move cursor ────────────────
        if landmarks:
            raw_x = np.interp(landmarks[8].x, [0.05, 0.95], [0, SCREEN_WIDTH])
            raw_y = np.interp(landmarks[8].y, [0.05, 0.95], [0, SCREEN_HEIGHT])
            sx, sy = self.get_smoothed_pos((raw_x, raw_y))
            pyautogui.moveTo(sx, sy, _pause=False)

        # ── Release drag if gesture changed ───
        if gesture not in (GestureState.LEFT_CLICK, GestureState.DRAGGING):
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False

        # ── Handle each gesture ───────────────

        # MOVE - just cursor movement (already done above)
        if gesture == GestureState.MOVE:
            self.last_action = gesture

        # LEFT_CLICK / DRAG (same pose, duration decides)
        elif gesture == GestureState.LEFT_CLICK:
            if self.last_action != GestureState.LEFT_CLICK:
                # Fresh detection - start timer
                self.gesture_start_time = now
                self.last_action = GestureState.LEFT_CLICK

            hold_duration = now - self.gesture_start_time

            if not self.is_dragging and hold_duration >= DRAG_HOLD_THRESHOLD:
                # Held long enough → start dragging
                pyautogui.mouseDown()
                self.is_dragging = True
                self.last_action = GestureState.DRAGGING

            elif not self.is_dragging and hold_duration >= CLICK_HOLD_THRESHOLD:
                # Short hold → click (only once)
                if self.last_action == GestureState.LEFT_CLICK:
                    pyautogui.click()
                    self.gesture_start_time = now + 999  # Block re-trigger
                    time.sleep(0.15)

        # RIGHT_CLICK
        elif gesture == GestureState.RIGHT_CLICK:
            if self.last_action != GestureState.RIGHT_CLICK:
                self.gesture_start_time = now
                self.last_action = GestureState.RIGHT_CLICK
            elif now - self.gesture_start_time >= CLICK_HOLD_THRESHOLD:
                pyautogui.rightClick()
                self.gesture_start_time = now + 999
                time.sleep(0.15)

        # SCROLL UP
        elif gesture == GestureState.SCROLL_UP:
            if now - self.last_scroll_time >= SCROLL_COOLDOWN:
                pyautogui.scroll(5)
                self.last_scroll_time = now
            self.last_action = gesture

        # SCROLL DOWN
        elif gesture == GestureState.SCROLL_DOWN:
            if now - self.last_scroll_time >= SCROLL_COOLDOWN:
                pyautogui.scroll(-5)
                self.last_scroll_time = now
            self.last_action = gesture

        # VOLUME UP
        elif gesture == GestureState.VOLUME_UP:
            if now - self.last_volume_time >= VOLUME_COOLDOWN:
                pyautogui.press('volumeup', presses=1)
                self.last_volume_time = now
            self.last_action = gesture

        # VOLUME DOWN
        elif gesture == GestureState.VOLUME_DOWN:
            if now - self.last_volume_time >= VOLUME_COOLDOWN:
                pyautogui.press('volumedown', presses=1)
                self.last_volume_time = now
            self.last_action = gesture

        # ZOOM IN
        elif gesture == GestureState.ZOOM_IN:
            if now - self.last_zoom_time >= ZOOM_COOLDOWN:
                pyautogui.hotkey('ctrl', '+')
                self.last_zoom_time = now
            self.last_action = gesture

        # ZOOM OUT
        elif gesture == GestureState.ZOOM_OUT:
            if now - self.last_zoom_time >= ZOOM_COOLDOWN:
                pyautogui.hotkey('ctrl', '-')
                self.last_zoom_time = now
            self.last_action = gesture

        # MINIMIZE
        elif gesture == GestureState.MINIMIZE:
            if self.last_action != GestureState.MINIMIZE:
                pyautogui.hotkey('win', 'd')
                self.last_action = GestureState.MINIMIZE
                time.sleep(0.5)

        return self.last_action

# ─────────────────────────────────────────────
# HUD Overlay
# ─────────────────────────────────────────────
def draw_hud(img, action_text, fingers, is_dragging, hold_progress=0.0):
    h, w = img.shape[:2]

    # Semi-transparent top bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # Action text
    color = (0, 255, 255)
    if "DRAG" in action_text:   color = (0, 0, 255)
    if "CLICK" in action_text:  color = (0, 255, 0)
    if "SCROLL" in action_text: color = (255, 165, 0)
    if "VOLUME" in action_text: color = (255, 0, 255)
    if "ZOOM" in action_text:   color = (255, 200, 0)

    cv2.putText(img, f'Action: {action_text}', (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Finger indicators
    finger_labels = ['T', 'I', 'M', 'R', 'P']
    for i, (label, state) in enumerate(zip(finger_labels, fingers)):
        x = 10 + i * 45
        dot_color = (0, 255, 0) if state else (80, 80, 80)
        cv2.circle(img, (x + 15, 70), 15, dot_color, -1)
        cv2.putText(img, label, (x + 8, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Hold progress bar (for click/drag)
    if hold_progress > 0:
        bar_w = int(250 * min(hold_progress, 1.0))
        cv2.rectangle(img, (10, 95), (260, 115), (50, 50, 50), -1)
        bar_color = (0, 0, 255) if hold_progress >= 1.0 else (0, 200, 100)
        cv2.rectangle(img, (10, 95), (10 + bar_w, 115), bar_color, -1)
        cv2.rectangle(img, (10, 95), (260, 115), (200, 200, 200), 1)
        cv2.putText(img, "Hold...", (270, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Drag indicator
    if is_dragging:
        cv2.putText(img, '🔴 DRAGGING', (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ESC hint
    cv2.putText(img, "ESC to exit | Move mouse to top-left to abort",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (150, 150, 150), 1)

    return img

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  ✋ Hand Gesture Controller")
    print("=" * 50)
    print("⚠️  This script controls your mouse & keyboard!")
    print("   Move mouse to TOP-LEFT corner to abort (failsafe)")
    print()

    try:
        for i in range(3, 0, -1):
            print(f"  Starting in {i}...", end='\r')
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Aborted.")
        return

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print("❌ Could not open camera!")
        return

    controller   = GestureController()
    frame_delay  = 1 / FRAME_RATE
    last_frame_t = 0

    print("\n✅ Gesture Control Active!")
    print("─" * 40)
    print("  Gesture          │ Action")
    print("─" * 40)
    print("  Index only       │ Move cursor")
    print("  Thumb+Index hold │ Click / Drag")
    print("  Thumb+Middle hold│ Right Click")
    print("  II (no thumb)    │ Scroll UP")
    print("  II + Thumb       │ Scroll DOWN")
    print("  III (no thumb)   │ Zoom OUT")
    print("  III + Thumb      │ Zoom IN")
    print("  Pinky only       │ Volume DOWN")
    print("  Thumb only       │ Volume UP")
    print("  Closed fist      │ Minimize all")
    print("─" * 40)
    print("  ESC to exit\n")

    try:
        while cap.isOpened():
            now = time.time()

            # FPS limiter
            if now - last_frame_t < frame_delay:
                continue
            last_frame_t = now

            ret, img = cap.read()
            if not ret:
                continue

            img     = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = controller.hands.process(img_rgb)

            action_text  = "No hand detected"
            fingers      = [False] * 5
            hold_progress = 0.0

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_lms  = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                hand_label = handedness.classification[0].label

                # Draw styled landmarks
                controller.mp_draw.draw_landmarks(
                    img, hand_lms,
                    controller.mp_hands.HAND_CONNECTIONS,
                    controller.mp_draw_styles.get_default_hand_landmarks_style(),
                    controller.mp_draw_styles.get_default_hand_connections_style()
                )

                landmarks = hand_lms.landmark
                fingers   = controller.count_fingers(landmarks, hand_label)
                gesture   = controller.classify_gesture(fingers)

                # Calculate hold progress for click/drag
                if gesture == GestureState.LEFT_CLICK:
                    elapsed       = now - controller.gesture_start_time
                    hold_progress = elapsed / DRAG_HOLD_THRESHOLD

                # Execute and get resulting action
                result_action = controller.execute_action(gesture, landmarks)
                action_text   = result_action.value

            # Draw HUD
            img = draw_hud(img, action_text, fingers,
                           controller.is_dragging, hold_progress)

            cv2.imshow("✋ Gesture Control", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        if controller.is_dragging:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Exited cleanly.")

if __name__ == "__main__":
    main()