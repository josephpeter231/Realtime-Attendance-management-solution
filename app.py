import math
import cv2
import os
import time
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, session
import threading
import json
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "attendance_system_secret_key"

# Create required directories
if not os.path.exists("static"):
    os.makedirs("static")
    
if not os.path.exists("static/attendance_data"):
    os.makedirs("static/attendance_data")

if not os.path.exists("static/known_faces"):
    os.makedirs("static/known_faces")

if not os.path.exists("static/temp"):
    os.makedirs("static/temp")

if not os.path.exists("static/leave_letters"):
    os.makedirs("static/leave_letters")

# Global variables
camera = None
camera_lock = threading.Lock()
frame = None
attendance_in_progress = False
attendance_results = {}
current_students = {}
attendance_data = {}
subjects = ["DA", "IoT", "NLP", "KRR", "STM"]  # List of available subjects

def load_students():
    """Load student data from files or create if not exists"""
    global current_students
    student_file = "static/students.json"
    
    # Create students.json if not exists
    if not os.path.exists(student_file):
        with open(student_file, 'w') as f:
            json.dump({}, f)
        return {}
    
    # Load student data
    try:
        with open(student_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_students():
    """Save student data to file"""
    with open("static/students.json", 'w') as f:
        json.dump(current_students, f, indent=4)

def load_attendance():
    """Load attendance data from file"""
    attendance_file = "static/attendance.json"
    
    # Create attendance.json if not exists
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            json.dump({}, f)
        return {}
    
    # Load attendance data
    try:
        with open(attendance_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_attendance():
    """Save attendance data to file"""
    with open("static/attendance.json", 'w') as f:
        json.dump(attendance_data, f, indent=4)

# Initialize data
current_students = load_students()
attendance_data = load_attendance()

# Face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """Detect faces in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def detect_known_faces(frame, faces):
    """Match detected faces with registered students"""
    attendance = {"present": [], "unknown": 0}
    
    # Skip recognition if no registered students
    if len(current_students) == 0:
        attendance["unknown"] = len(faces)
        return attendance
    
    # Load all known faces
    known_face_images = {}
    for student_id in current_students:
        img_path = os.path.join("static/known_faces", f"student{student_id}.jpg")
        if os.path.exists(img_path):
            known_face_images[student_id] = cv2.imread(img_path)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Compare with each known face
        best_match = None
        best_score = float('inf')
        
        for student_id, known_img in known_face_images.items():
            try:
                # Resize known image to match detected face size
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
        if best_match and best_score < 10000000: 
            if best_match not in attendance["present"]:
                attendance["present"].append(best_match)
        else:
            attendance["unknown"] += 1
    
    return attendance

def gen_frames():
    """Generator to stream webcam frames"""
    global camera, frame
    
    if camera is None:
        with camera_lock:
            camera = cv2.VideoCapture(0)
            
    while True:
        success, new_frame = camera.read()
        if not success:
            break
        else:
            frame = new_frame.copy()
            
            # Detect faces and draw rectangles
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            # Display face count
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Modify the take_attendance_snapshots function to accept the subject as a parameter
def take_attendance_snapshots(subject=None):
    """Take 4 snapshots at 10-second intervals and process attendance"""
    global attendance_in_progress, attendance_results, frame
    
    if attendance_in_progress:
        return {"error": "Attendance already in progress"}
    
    # Use the provided subject or default to "Unknown"
    if subject not in subjects:
        subject = "Unknown"
    
    attendance_in_progress = True
    snapshot_count = 4
    interval = 10  # seconds
    attendance_results = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "subject": subject,
        "snapshots": [],
        "student_attendance": {},
        "total_registered": len(current_students)
    }
    
    # Initialize attendance for each student (0%)
    for student_id in current_students:
        attendance_results["student_attendance"][student_id] = {
            "name": current_students[student_id]["name"],
            "present_count": 0,
            "percentage": 0
        }
    
    for i in range(snapshot_count):
        # Sleep for interval seconds (except first snapshot)
        if i > 0:
            time.sleep(interval)
        
        # Take snapshot
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"snapshot_{timestamp}_{i+1}.jpg"
            snapshot_path = os.path.join("static/attendance_data", snapshot_filename)
            
            # Save the snapshot
            cv2.imwrite(snapshot_path, frame)
            
            # Process faces in snapshot
            faces = detect_faces(frame)
            attendance = detect_known_faces(frame, faces)
            
            # Record present students
            snapshot_data = {
                "filename": snapshot_filename,
                "present": attendance["present"],
                "unknown": attendance["unknown"]
            }
            attendance_results["snapshots"].append(snapshot_data)
            
            # Update student attendance count
            for student_id in attendance["present"]:
                if student_id in attendance_results["student_attendance"]:
                    attendance_results["student_attendance"][student_id]["present_count"] += 1
    
    # Calculate attendance percentages (25% for each snapshot present)
    for student_id in attendance_results["student_attendance"]:
        present_count = attendance_results["student_attendance"][student_id]["present_count"]
        percentage = present_count * 25  # 25% per snapshot
        attendance_results["student_attendance"][student_id]["percentage"] = percentage
    
    # Save attendance record
    date_key = datetime.now().strftime("%Y-%m-%d")
    time_key = datetime.now().strftime("%H:%M:%S")
    
    if date_key not in attendance_data:
        attendance_data[date_key] = {}
    
    if subject not in attendance_data[date_key]:
        attendance_data[date_key][subject] = {}
    
    attendance_data[date_key][subject][time_key] = attendance_results
    save_attendance()
    
    attendance_in_progress = False
    return attendance_results

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        roll_no = request.form['roll_no']
        
        # Generate unique ID
        student_id = str(len(current_students) + 1)
        while student_id in current_students:
            student_id = str(int(student_id) + 1)
        
        # Create student record
        current_students[student_id] = {
            "id": student_id,
            "name": name,
            "roll_no": roll_no,
            "registered_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_students()
        
        # Store the ID for the capture phase
        return render_template('capture.html', student_id=student_id, student_name=name)
    
    return render_template('register.html')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    data = request.get_json()
    student_id = data.get('student_id')
    image_data = data.get('image')
    
    if not student_id or not image_data:
        return jsonify({"success": False, "message": "Missing student ID or image data"})
    
    # Convert base64 to image
    import base64
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Save the image
    filename = f"student{student_id}.jpg"
    filepath = os.path.join("static/known_faces", filename)
    
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    return jsonify({"success": True, "filename": filename})

@app.route('/students')
def students():
    return render_template('students.html', students=current_students)

@app.route('/take_attendance')
def take_attendance():
    # Store the selected subject in session if provided
    if 'subject' in request.args:
        session['current_subject'] = request.args.get('subject')
    
    return render_template('take_attendance.html', 
                          subjects=subjects, 
                          current_subject=session.get('current_subject'))

# Modify the start_attendance route to pass the subject to the thread
@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    # Get subject from form data
    subject = request.form.get('subject')
    if subject not in subjects:
        subject = "Unknown"
    
    # Store in session for UI consistency
    session['current_subject'] = subject
    
    # Pass the subject to the thread
    thread = threading.Thread(target=take_attendance_snapshots, args=(subject,))
    thread.start()
    
    return jsonify({"success": True, "message": "Attendance process started", "subject": subject})

@app.route('/attendance_status')
def attendance_status():
    return jsonify({
        "in_progress": attendance_in_progress,
        "results": attendance_results if not attendance_in_progress and attendance_results else None
    })

@app.route('/attendance_report')
def attendance_report():
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    
    # Get available dates
    dates = list(attendance_data.keys())
    
    # Get subjects for selected date
    subjects_for_date = []
    if date in attendance_data:
        subjects_for_date = list(attendance_data[date].keys())
    
    # Get selected subject
    selected_subject = request.args.get('session')
    
    # Get sessions (times) for selected date and subject
    sessions = []
    if date in attendance_data and selected_subject in attendance_data[date]:
        sessions = list(attendance_data[date][selected_subject].keys())
    
    # Get session time if specified
    session_time = request.args.get('time')
    
    # Get session data if all parameters are specified
    session_data = None
    if (date in attendance_data and 
        selected_subject in attendance_data[date] and 
        session_time in attendance_data[date][selected_subject]):
        session_data = attendance_data[date][selected_subject][session_time]
    # If session_time not specified but only one exists, use it
    elif (date in attendance_data and 
          selected_subject in attendance_data[date] and
          len(attendance_data[date][selected_subject]) == 1):
        session_time = list(attendance_data[date][selected_subject].keys())[0]
        session_data = attendance_data[date][selected_subject][session_time]
    
    return render_template('attendance_report.html', 
                         dates=dates, 
                         selected_date=date,
                         subjects=subjects_for_date,
                         selected_subject=selected_subject,
                         sessions=sessions,
                         selected_time=session_time,
                         session_data=session_data,
                         all_subjects=subjects,
                         students=current_students)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global camera
    if camera:
        with camera_lock:
            camera.release()
            camera = None
    
    # Clean up resources and save data
    save_students()
    save_attendance()
    
    return jsonify({"success": True})

# Add function to load leave data
def load_leave_data():
    """Load leave data from file"""
    leave_file = "static/leave_data.json"
    
    # Create leave_data.json if not exists
    if not os.path.exists(leave_file):
        with open(leave_file, 'w') as f:
            json.dump({}, f)
        return {}
    
    # Load leave data
    try:
        with open(leave_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_leave_data(leave_data):
    """Save leave data to file"""
    with open("static/leave_data.json", 'w') as f:
        json.dump(leave_data, f, indent=4)

# Initialize leave data
leave_data = load_leave_data()

# Add the route for leave application
@app.route('/leave', methods=['GET', 'POST'])
def leave_application():
    global leave_data
    
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        reason = request.form.get('reason')
        
        if not student_id or student_id not in current_students:
            flash('Invalid student selected', 'danger')
            return redirect(url_for('leave_application'))
            
        # Check if PDF was uploaded
        if 'leave_letter' not in request.files:
            flash('No leave letter uploaded', 'danger')
            return redirect(url_for('leave_application'))
            
        file = request.files['leave_letter']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('leave_application'))
            
        if file and file.filename.lower().endswith('.pdf'):
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = secure_filename(f"leave_{student_id}_{timestamp}.pdf")
            file_path = os.path.join("static/leave_letters", filename)
            
            # Save the file
            file.save(file_path)
            
            # Create leave record
            leave_id = timestamp
            
            if student_id not in leave_data:
                leave_data[student_id] = {}
                
            leave_data[student_id][leave_id] = {
                "from_date": from_date,
                "to_date": to_date,
                "reason": reason,
                "file": filename,
                "status": "pending",  # pending, approved, rejected
                "submitted_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "comments": ""
            }
            
            save_leave_data(leave_data)
            flash('Leave application submitted successfully!', 'success')
            return redirect(url_for('leave_status'))
        else:
            flash('Only PDF files are allowed', 'danger')
            return redirect(url_for('leave_application'))
    
    return render_template('leave_form.html', students=current_students)

@app.route('/leave_status')
def leave_status():
    student_id = request.args.get('student_id', '')
    
    # Filter leave applications
    filtered_leaves = {}
    for sid, leaves in leave_data.items():
        if not student_id or sid == student_id:
            filtered_leaves[sid] = leaves
            
    return render_template('leave_status.html', 
                          leave_data=filtered_leaves, 
                          students=current_students,
                          selected_student=student_id)

@app.route('/review_leave/<student_id>/<leave_id>', methods=['GET', 'POST'])
def review_leave(student_id, leave_id):
    if student_id not in leave_data or leave_id not in leave_data[student_id]:
        flash('Leave application not found', 'danger')
        return redirect(url_for('leave_status'))
        
    if request.method == 'POST':
        status = request.form.get('status')
        comments = request.form.get('comments', '')
        
        if status in ['approved', 'rejected']:
            leave_data[student_id][leave_id]['status'] = status
            leave_data[student_id][leave_id]['comments'] = comments
            leave_data[student_id][leave_id]['reviewed_on'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            save_leave_data(leave_data)
            flash(f'Leave application {status.capitalize()}', 'success')
            return redirect(url_for('leave_status'))
        else:
            flash('Invalid status', 'danger')
    
    leave_info = leave_data[student_id][leave_id]
    student_name = current_students[student_id]['name'] if student_id in current_students else 'Unknown Student'
    
    return render_template('review_leave.html', 
                          student_id=student_id,
                          student_name=student_name,
                          leave_id=leave_id,
                          leave_info=leave_info)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', students=current_students)

@app.route('/student_report/<student_id>')
def student_report(student_id):
    if student_id not in current_students:
        flash('Student not found', 'danger')
        return redirect(url_for('dashboard'))
    
    # Generate report for last 15 days by default
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15)
    
    # Format dates for display
    end_date_str = end_date.strftime("%d-%m-%Y")
    start_date_str = start_date.strftime("%d-%m-%Y")
    
    # Get dates within the range
    report_dates = []
    for date_str in attendance_data.keys():
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if start_date <= date_obj <= end_date:
                report_dates.append(date_str)
        except ValueError:
            continue
    
    # Prepare report data
    subject_attendance = {}
    for subject in subjects:
        subject_attendance[subject] = {
            "total_sessions": 0,
            "present_sessions": 0,
            "percentage": 0
        }
    
    # Calculate attendance for each subject
    for date_str in report_dates:
        for subject in subjects:
            if date_str in attendance_data and subject in attendance_data[date_str]:
                sessions = attendance_data[date_str][subject]
                for time_key, session_data in sessions.items():
                    subject_attendance[subject]["total_sessions"] += 1
                    
                    if (student_id in session_data["student_attendance"] and 
                        session_data["student_attendance"][student_id]["percentage"] >= 75):
                        subject_attendance[subject]["present_sessions"] += 1
    
    # Calculate percentages and status
    for subject in subjects:
        total = subject_attendance[subject]["total_sessions"]
        present = subject_attendance[subject]["present_sessions"]
        
        if total > 0:
            percentage = round((present / total) * 100)
        else:
            percentage = 0
            
        subject_attendance[subject]["percentage"] = percentage
        
        # Determine status and advice
        if percentage >= 85:
            subject_attendance[subject]["status"] = "Excellent"
            subject_attendance[subject]["icon"] = "✅"
            subject_attendance[subject]["advice"] = ""
        elif percentage >= 80:
            subject_attendance[subject]["status"] = "Good"
            subject_attendance[subject]["icon"] = "✅"
            subject_attendance[subject]["advice"] = ""
        elif percentage >= 75:
            subject_attendance[subject]["status"] = "Satisfactory"
            subject_attendance[subject]["icon"] = "✅"
            subject_attendance[subject]["advice"] = ""
        elif percentage >= 70:
            subject_attendance[subject]["status"] = "Need Improvement"
            subject_attendance[subject]["icon"] = "⚠"
            sessions_needed = math.ceil(max(0, (0.75 * total - present) / 0.75))
            subject_attendance[subject]["advice"] = f"Attend {sessions_needed} more sessions"
        else:
            subject_attendance[subject]["status"] = "At Risk"
            subject_attendance[subject]["icon"] = "❗"
            sessions_needed = math.ceil(max(0, (0.75 * total - present) / 0.75))
            subject_attendance[subject]["advice"] = f"Attend {sessions_needed} more sessions urgently"
    
    # Generate summary and advice
    risk_subjects = [s for s in subjects if subject_attendance[s]["percentage"] < 75]
    improvement_subjects = [s for s in subjects if 70 <= subject_attendance[s]["percentage"] < 75]
    good_subjects = [s for s in subjects if subject_attendance[s]["percentage"] >= 75]
    
    return render_template('student_report.html',
                          student=current_students[student_id],
                          subject_attendance=subject_attendance,
                          start_date=start_date_str,
                          end_date=end_date_str,
                          risk_subjects=risk_subjects,
                          improvement_subjects=improvement_subjects,
                          good_subjects=good_subjects,
                          report_date=datetime.now().strftime("%d-%m-%Y"),
                          subjects=subjects)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)