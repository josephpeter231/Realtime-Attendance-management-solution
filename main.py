import cv2
import os
import time
import numpy as np
from datetime import datetime

# Create directories for storing attendance data if they don't exist
if not os.path.exists("attendance_data"):
    os.makedirs("attendance_data")

if not os.path.exists("known_faces"):
    os.makedirs("known_faces")
    print("Please add reference images of known students to the 'known_faces' folder")

# Use 0 for webcam (you can try other integers if you have multiple cameras)
cap = cv2.VideoCapture(0)  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load known student faces
known_face_images = {}
student_names = []
print("Loading registered students...")
for file in os.listdir("known_faces"):
    if file.startswith("student") and (file.endswith(".jpg") or file.endswith(".png")):
        student_id = file.replace("student", "").split(".")[0]
        student_path = os.path.join("known_faces", file)
        student_img = cv2.imread(student_path)
        if student_img is not None:
            known_face_images[student_id] = student_img
            student_names.append(f"Student {student_id}")
            print(f"Loaded {file}")

total_students = len(known_face_images)
if total_students == 0:
    print("No registered students found. Please run register_students.py first.")

# Track attendance timing
last_capture_time = time.time()
capture_interval = 10  # Take snapshot every 60 seconds (1 minute)
attendance_record = {}
attendance_count = 0

print("Press 'q' to quit")
print("System will capture attendance every minute")

def detect_known_faces(frame, faces):
    """
    Face recognition for attendance by comparing detected faces with known student faces
    
    Returns: Dictionary with counts of recognized faces and list of identified students
    """
    attendance = {"Present": 0, "Unknown": 0}
    present_students = []
    
    # Skip recognition if no registered students
    if total_students == 0:
        attendance["Present"] = len(faces)
        return attendance
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # For demo, use a simple comparison method (in practice, use a proper face recognition library)
        best_match = None
        best_score = float('inf')
        
        # Compare with each known face using a simple image difference method
        # Note: This is a basic demonstration and not reliable for real-world use
        for student_id, known_img in known_face_images.items():
            try:
                # Resize known image to match detected face size for comparison
                resized_known = cv2.resize(known_img, (w, h))
                
                # Convert both to grayscale for comparison
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray_known = cv2.cvtColor(resized_known, cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference between images
                diff = cv2.absdiff(gray_face, gray_known)
                score = np.sum(diff)
                
                if score < best_score:
                    best_score = score
                    best_match = student_id
            except Exception as e:
                print(f"Error comparing faces: {e}")
        
        # Use a threshold to determine if the face matches a known student
        # This threshold needs tuning based on testing
        if best_match and best_score < 10000000:  # Threshold needs adjustment
            attendance["Present"] += 1
            if best_match not in present_students:
                present_students.append(best_match)
        else:
            attendance["Unknown"] += 1
    
    attendance["present_students"] = present_students
    
    return attendance

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Take snapshot and record attendance every minute
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the snapshot
        filename = os.path.join("attendance_data", f"attendance_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        # Record attendance
        attendance = detect_known_faces(frame, faces)
        attendance_record[timestamp] = attendance
        
        # Calculate attendance percentage if registered students exist
        if total_students > 0:
            attendance_percentage = (len(attendance.get("present_students", [])) / total_students) * 100
        else:
            attendance_percentage = 0
            
        attendance_count += 1
        print(f"\n--- ATTENDANCE RECORD {timestamp} ---")
        print(f"Students present: {attendance['Present']}")
        
        # List recognized students
        if "present_students" in attendance and attendance["Present"] > 0:
            print("Recognized students:")
            for student_id in attendance["present_students"]:
                print(f"- Student {student_id}")
        
        print(f"Unknown faces: {attendance['Unknown']}")
        print(f"Attendance snapshot saved as {filename}")
        
        if total_students > 0:
            print(f"Current attendance percentage: {attendance_percentage:.1f}%")
        
        last_capture_time = current_time
    
    # Display the number of faces detected
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display time until next capture
    time_remaining = int(capture_interval - (current_time - last_capture_time))
    cv2.putText(frame, f"Next capture in: {time_remaining}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display student info
    cv2.putText(frame, f"Students: {total_students}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display attendance count
    cv2.putText(frame, f"Records: {attendance_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Attendance System', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance record to a file
with open(os.path.join("attendance_data", "attendance_log.txt"), "w") as f:
    f.write(f"Attendance Log - {datetime.now().strftime('%Y-%m-%d')}\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total registered students: {total_students}\n\n")
    for timestamp, data in attendance_record.items():
        present = data.get("Present", 0)
        students_list = ", ".join(data.get("present_students", []))
        f.write(f"Time: {timestamp} | Present: {present}")
        if students_list:
            f.write(f" | Students: {students_list}")
        f.write("\n")

cap.release()
cv2.destroyAllWindows()