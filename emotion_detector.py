import cv2
from deepface import DeepFace

def main():
    print("Initializing Emotion Detection...")
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        try:
            # Analyze the frame for emotions using deepface
            # enforce_detection=False prevents crashes when no face is detected
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # Handle both dictionary and list returns depending on deepface version
            if isinstance(results, dict):
                results = [results]
                
            for result in results:
                emotion = result.get('dominant_emotion', 'Unknown')
                
                # Try to get the bounding box region
                region = result.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                
                if w > 0 and h > 0:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Put the emotion text above the face
                    cv2.putText(frame, emotion.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, emotion.capitalize(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
        except Exception as e:
            # If there's an error in analysis, just show the frame
            pass 

        # Display the resulting frame
        cv2.imshow('Emotion Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
