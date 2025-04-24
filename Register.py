import cv2
import os
import time
import numpy as np

# Create directory for known faces if it doesn't exist
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

def register_new_student():
    """Register a new student by capturing their face"""
    # Find the next available student number
    existing_students = [f for f in os.listdir("known_faces") if f.startswith("student")]
    next_number = 1
    if existing_students:
        numbers = [int(name.replace("student", "").split(".")[0]) for name in existing_students]
        next_number = max(numbers) + 1
    
    student_filename = f"student{next_number}.jpg"
    
    print(f"\nRegistering new student (ID: {next_number})")
    print("Position your face in the camera frame")
    print("Press 'c' to capture image when ready")
    print("Press 'q' to cancel")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Countdown timer variables
    countdown_active = False
    countdown_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces
        face_found = False
        for (x, y, w, h) in faces:
            face_found = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Position face in frame and press 'c'", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        # Show countdown if active
        if countdown_active:
            current_time = time.time()
            remaining = int(countdown_time - current_time)
            
            if remaining <= 0:
                # Capture final image with face
                if face_found:
                    # Save the largest face for better recognition
                    largest_face = None
                    largest_area = 0
                    
                    for (x, y, w, h) in faces:
                        if w * h > largest_area:
                            largest_area = w * h
                            largest_face = (x, y, w, h)
                    
                    if largest_face:
                        x, y, w, h = largest_face
                        # Add a margin around the face
                        margin = 30
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(frame.shape[1] - x, w + 2 * margin)
                        h = min(frame.shape[0] - y, h + 2 * margin)
                        
                        # Extract face with margin
                        face_img = frame[y:y+h, x:x+w]
                        face_path = os.path.join("known_faces", student_filename)
                        cv2.imwrite(face_path, face_img)
                        print(f"Student registered successfully as {student_filename}!")
                    
                    countdown_active = False
                    break
                else:
                    print("No face detected, please try again.")
                    countdown_active = False
            
            # Display countdown
            cv2.putText(frame, f"Capturing in: {remaining}s", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Student Registration', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Registration canceled")
            break
        elif key == ord('c') and not countdown_active:
            if face_found:
                countdown_active = True
                countdown_time = time.time() + 3  # 3 second countdown
            else:
                print("No face detected. Please position your face in the frame.")
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    return face_found

def main():
    while True:
        print("\n==== STUDENT FACE REGISTRATION SYSTEM ====")
        print("1. Register a new student")
        print("2. View registered students")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            register_new_student()
        elif choice == '2':
            students = [f for f in os.listdir("known_faces") if f.startswith("student")]
            if students:
                print("\nRegistered students:")
                for student in sorted(students):
                    student_id = student.replace("student", "").split(".")[0]
                    print(f"- Student ID: {student_id} (File: {student})")
            else:
                print("No students registered yet.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()