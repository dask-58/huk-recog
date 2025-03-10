from flask import Flask, jsonify
import os
import cv2
import face_recognition
import requests
import json
import base64
import ast
import numpy as np
from datetime import datetime
import psycopg2
from dotenv import load_dotenv
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

app = Flask(__name__)

def poll_images(num_images=4):
    UPLOADTHING_TOKEN = 'eyJhcGlLZXkiOiJza19saXZlXzkwMmE4YmE3YWM0MDIzZmFlNWE2YWI0OWUzNDgyNWQ3NzMzZWE3M2NmMmIwZTY0N2NhNGE1NDgxM2ViOWQ0NDgiLCJhcHBJZCI6Imk3eHFheHJuN2IiLCJyZWdpb25zIjpbInNlYTEiXX0='
    token_data = base64.b64decode(UPLOADTHING_TOKEN).decode('utf-8')
    token_dict = ast.literal_eval(token_data)
    APP_ID = token_dict.get('appId', 'i7xqaxrn7b')

    headers = {
        "X-Uploadthing-Api-Key": "sk_live_902a8ba7ac4023fae5a6ab49e34825d7733ea73cf2b0e647ca4a54813eb9d448",
        "Content-Type": "application/json"
    }

    try:
        print("Fetching latest images...")
        list_response = requests.post(
            "https://api.uploadthing.com/v6/listFiles",
            headers=headers,
            json={
                "limit": num_images,
                "sort": "uploadedAt",
                "order": "desc"
            }
        )
        list_response.raise_for_status()
        files_data = list_response.json()
        files_list = files_data.get("files", [])
        if not files_list:
            print("No files found in response.")
            return []
        
        image_urls = []
        for file_data in files_list:
            file_key = file_data["key"]
            print(f"Polling for file: {file_key}...")
            poll_response = requests.get(
                f"https://api.uploadthing.com/v6/pollUpload/{file_key}",
                headers=headers
            )
            poll_response.raise_for_status()
            poll_data = poll_response.json()
            if "fileUrl" in poll_data:
                file_url = poll_data["fileUrl"]
            else:
                file_url = f"https://{APP_ID}.ufs.sh/f/{file_key}"
                # print(f"Warning: fileUrl not found in response. Constructed URL: {file_url}")
            image_urls.append(file_url)
            print(f"  - URL: {file_url}")
        
        return image_urls

    except Exception as e:
        print("Error fetching images:", e)
        return []

def recognize_faces(image_urls, known_encodings, known_names):
    aggregated_detected_names = []
    aggregated_total_faces = 0

    for idx, file_url in enumerate(image_urls, start=1):
        try:
            image_response = requests.get(file_url)
            image_response.raise_for_status()
        except Exception as e:
            print(f"Error fetching image {file_url}: {e}")
            continue

        image_array = np.frombuffer(image_response.content, np.uint8)
        test_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if test_image is None:
            print(f"Error decoding image from {file_url}")
            continue

        rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_test_image, model="hog")
        aggregated_total_faces += len(face_locations)
        test_encodings = face_recognition.face_encodings(rgb_test_image, face_locations)

        detected_names = []
        THRESHOLD = 0.55
        for encoding in test_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            min_distance = np.min(distances) if len(distances) > 0 else 1.0
            if min_distance < THRESHOLD:
                best_match_idx = int(np.argmin(distances))
                name = known_names[best_match_idx]
                try:
                    last_two_digits = str(int(name[5:]))
                except:
                    last_two_digits = name
                detected_names.append(last_two_digits)
            else:
                detected_names.append("Unknown")
        aggregated_detected_names.extend(detected_names)

    attendance = {
        "timestamp": datetime.now().isoformat(),
        "detected_names": aggregated_detected_names,
        "total_faces": aggregated_total_faces
    }
    return attendance

def update_attendance_in_db(attendance):
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL not found in environment variables")
        return

    timestamp = attendance.get("timestamp")
    detected_rolls = attendance.get("detected_names", [])
    
    if not timestamp:
        print("No timestamp found in attendance data")
        return

    dt = datetime.fromisoformat(timestamp)
    date_field = f"date_{dt.year}_{dt.month:02d}_{dt.day:02d}"
    print(f"Updating attendance for date field: {date_field}")
    
    try:
        # Convert valid roll strings to integers (roll numbers)
        present_rolls = [int(roll) for roll in detected_rolls if roll.isdigit()]
    except Exception as e:
        print("Error converting roll numbers to integers:", e)
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print("Error connecting to database:", e)
        return

    try:
        with conn.cursor() as cur:
            # Update the Attendance table for today's class.
            attendance_query = f'''
                UPDATE "Attendance" 
                SET {date_field} = CASE 
                    WHEN "studentRollNo" = ANY(%s) THEN %s::"AttendanceStatus" 
                    ELSE %s::"AttendanceStatus" 
                END
            '''
            cur.execute(attendance_query, (present_rolls, "Present", "Absent"))

            # Update attendancePercentage column.
            attendance_percentage_query = '''
                UPDATE "Attendance"
                SET "attendancePercentage" =
                    CASE
                      WHEN total_classes = 0 THEN 0
                      ELSE present_count * 100.0 / total_classes
                    END
                FROM (
                    SELECT "studentRollNo",
                        (
                            (CASE WHEN "date_2025_03_01" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_02" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_03" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_04" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_05" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_06" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_07" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_08" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_09" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_10" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_11" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_12" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_13" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_14" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_15" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_16" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_17" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_18" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_19" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_20" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_21" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_22" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_23" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_24" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_25" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_26" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_27" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_28" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_29" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_30" IS NOT NULL THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_31" IS NOT NULL THEN 1 ELSE 0 END)
                        ) AS total_classes,
                        (
                            (CASE WHEN "date_2025_03_01" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_02" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_03" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_04" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_05" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_06" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_07" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_08" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_09" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_10" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_11" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_12" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_13" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_14" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_15" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_16" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_17" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_18" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_19" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_20" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_21" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_22" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_23" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_24" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_25" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_26" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_27" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_28" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_29" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_30" = 'Present' THEN 1 ELSE 0 END) +
                            (CASE WHEN "date_2025_03_31" = 'Present' THEN 1 ELSE 0 END)
                        ) AS present_count
                    FROM "Attendance"
                ) sub
                WHERE "Attendance"."studentRollNo" = sub."studentRollNo";
            '''
            cur.execute(attendance_percentage_query)

        conn.commit()
        print("Attendance update completed.")
        print("Attendance percentage update completed.")
    except Exception as e:
        conn.rollback()
        print("Error occurred while updating attendance:", e)
    finally:
        conn.close()

def send_absence_emails(attendance_data):
    sender_email = "badam152.hukum@gmail.com"
    sender_password = "atskhzvcchkgdnpb"
    
    # Get present roll numbers from attendance data
    present_rolls = set(str(roll) for roll in attendance_data.get("detected_names", []) if str(roll).isdigit())
    
    # Generate all valid roll numbers (excluding 14 and 53)
    all_roll_numbers = {str(i).zfill(3) for i in range(1, 71)} - {'014', '053'}
    
    # Find absentees
    absent_rolls = all_roll_numbers - present_rolls
    
    for roll in absent_rolls:
        receiver_email = f"23bcs{roll}@iiitdwd.ac.in"
        
        subject = "TEST Alert from HUKUM: Absence Notification"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <p>Dear <b>23BCS{roll}</b>,</p>
            <p>We noticed that you were marked <b style="color:red;">absent</b> for today's class session. Regular attendance is crucial for your academic progress.</p>
            <p>If you were <b style="color:green;">present</b> in the class, we are extremely sorry (machines make mistakes). Please inform the discrepancy to the respected faculty.</p>
            <br>
            <p>Best Regards,</p>
            <p><b>Faculty, Hukum Organization</b></p>
            <p style="font-size: 12px; color: gray;">This is an automated email. Please do not reply.</p>
            <p style="font-size: 12px; color: gray;">This is a test from HUKUM. PLEASE ignore.</p>
        </body>
        </html>
        """
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print(f"✅ Email sent to {receiver_email}")
        except Exception as e:
            print(f"❌ Failed to send email to {receiver_email}: {e}")

@app.route('/update_attendance', methods=['GET'])
def update_attendance_route():
    try:
        # Load known faces and embeddings
        print("Loading encodings...")
        with open("data/encodings.pkl", "rb") as file:
            data = pickle.load(file)
        known_encodings = data["encodings"]
        known_names = data["names"]

        # Poll for latest images
        image_urls = poll_images()
        if not image_urls:
            return jsonify({"error": "No images to process."}), 404

        # Recognize faces and generate attendance data
        attendance = recognize_faces(image_urls, known_encodings, known_names)

        # Update attendance in the database
        update_attendance_in_db(attendance)

        # Send emails to absentees
        # send_absence_emails(attendance)

        # Return the attendance data as a response
        return jsonify(attendance)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app in debug mode (for development)
    app.run(host="0.0.0.0", debug=True, port=5001)
