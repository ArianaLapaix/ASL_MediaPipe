import cv2
import mediapipe as mp 
import numpy as np 

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# Landmark Labels - List of integers representing the 21 landmarks
# These are the indices for each landmark in the MediaPipe Hands model
# 0: Wrist
# 1-4: Thumb
# 5-8: Index Finger
# 9-12: Middle Finger
# 13-16: Ring Finger
# 17-20: Pinky Finger
landmark_labels = [
    "wrist",
    "thumb_CMC",
    "thumb_MCP",
    "thumb_IP",
    "thumb_tip",
    "index_finger_CMC",
    "index_finger_MCP",
    "index_finger_PIP",
    "index_finger_TIP",
    "middle_finger_CMC",
    "middle_finger_MCP",
    "middle_finger_PIP",
    "middle_finger_TIP",
    "ring_finger_CMC",
    "ring_finger_MCP",
    "ring_finger_PIP",
    "ring_finger_TIP",
    "pinky_finger_CMC",
    "pinky_finger_MCP",
    "pinky_finger_PIP",
    "pinky_finger_TIP",
]


def extract_hand_landmarks(results):
    """Extract hand landmarks from MediaPipe Hands results"""
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

def normalize_to_wrist(landmarks):
    """Normalize landmarks to be between 0 and 1"""
    reshaped = landmarks.reshape(21, 3)
    wrist = reshaped[0].copy()
    reshaped -= wrist
    return reshaped.flatten()  
    
# Main detection loop

print("Starting hand detection...\n")
print("Press 's' to save a hand landmark set, or 'q' to quit.")

# Initialize MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip for selfie view (mirror image)
        image = cv2.flip(image, 1)

        # Convert BGR -> RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False

        # Run Detection
        results = hands.process(rgb_image)

        # Draw results
        rgb_image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        landmark_data = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract Landmark Data 
                raw_landmarks = extract_hand_landmarks(results)   
                
                # Normalize Landmark Data 
                landmark_data = normalize_to_wrist(raw_landmarks)

                # Show landmark count on screen
                cv2.putText(image, f"Landmarks: {len(landmark_data) // 3}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            

        # Display the image
        cv2.imshow("MediaPipe Hands", image)


        # Add keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break    
        if key == ord("s") and landmark_data is not None:
            # Save landmark data
            np.save(f"data/hand_landmarks_{time.time()}.npy", landmark_data)     

cap.release()
cv2.destroyAllWindows()
print("Done.")
                