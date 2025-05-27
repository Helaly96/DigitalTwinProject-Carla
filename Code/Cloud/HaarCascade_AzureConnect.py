import cv2
import numpy as np
from datetime import datetime
import socket
import ssl
import pytds
import base64

# Azure SQL connection
server = 'iotservertest2.database.windows.net'
port = 1433
database = 'iotproject_dev'
username = 'azureuser@iotservertest2'
password = '12345678a@'

def get_db_connection():
    sock = socket.create_connection((server, port))
    context = ssl.create_default_context()
    secure_sock = context.wrap_socket(sock, server_hostname=server)
    conn = pytds.connect(
        sock=secure_sock,
        server=server,
        database=database,
        user=username,
        password=password,
    )
    return conn

def ensure_table(cursor):
    cursor.execute("""
    IF OBJECT_ID('SmileEvents', 'U') IS NULL
    CREATE TABLE SmileEvents (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        Timestamp DATETIME,
        Sensor NVARCHAR(50),
        Event NVARCHAR(50),
        ImageBase64 NVARCHAR(MAX)
    );
    """)

def frame_to_base64(frame):
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        return None
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64

def insert_detection(conn, timestamp, sensor, event, image_b64):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO SmileEvents (Timestamp, Sensor, Event, ImageBase64) VALUES (%s, %s, %s, %s)",
        (timestamp, sensor, event, image_b64)
    )
    conn.commit()
    cursor.close()

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

conn = get_db_connection()
ensure_table(conn.cursor())

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'SMILE', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            img_b64 = frame_to_base64(frame)
            if img_b64:
                insert_detection(conn, timestamp, "Camera_1", "SMILE", img_b64)
                print(f"😄 Smile saved at {timestamp}")

    cv2.imshow("Smile Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
