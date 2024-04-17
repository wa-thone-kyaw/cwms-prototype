from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
import threading
import winsound
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Load YOLO
net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# Path to your video file
video_path = "need.mp4"


# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Initialize a dictionary to store detected people and their detection time
detected_people = {}

# Initialize a counter to assign unique IDs to new people
person_id_counter = 1


# Frame resizing parameters
frame_width = 640
frame_height = 480
setTime = 10


@app.route("/stop_all_alarms", methods=["POST"])
def stop_all_alarms():
    for person_info in detected_people.values():
        person_info["alarm_triggered"] = True  # Change True to False
    return jsonify({"message": "All alarms stopped"}), 200


# @app.route("/stop_alarm/<int:person_id>", methods=["POST"])
# def stop_alarm(person_id):
#     if person_id in detected_people:
#         detected_people[person_id]["alarm_triggered"] = True
#         return jsonify({"message": f"Alarm stopped for person {person_id}"}), 200
#     else:
#         return jsonify({"message": "Person not found"}), 404


# Customizable settings
@app.route("/settings", methods=["GET", "POST"])
def settings():
    global setTime
    if request.method == "POST":
        data = request.json
        setTime = float(data.get("setTime", setTime))  # Convert setTime to float
    return jsonify({"setTime": setTime})


def update_waiting_times():
    while True:
        current_time = time.time()
        for person_id, person_info in list(detected_people.items()):
            elapsed_time = current_time - person_info["time"]
            person_info["time"] = person_info["time"] + (
                current_time - person_info["last_update_time"]
            )
            person_info["last_update_time"] = current_time

            # if elapsed_time > setTime and not person_info.get("alarm_triggered", False):
            #   person_info["alarm_triggered"] = True
            # winsound.Beep(1000, 500)  # Play alarm sound (1000 Hz for 500 ms)
            # time.sleep(3)

        socketio.emit("update_waiting_times", get_waiting_times())
        time.sleep(1)  # Update waiting times every 1 second


def get_waiting_times():
    waiting_times = {}
    current_time = time.time()
    for person_id, person_info in detected_people.items():
        elapsed_time = current_time - person_info["time"]
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        waiting_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        if elapsed_time > setTime:
            waiting_time_str += " (EXCEEDED)"
        waiting_times[f"Person {person_info['id']}"] = waiting_time_str
    return waiting_times


update_thread = threading.Thread(target=update_waiting_times)
update_thread.daemon = True
update_thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        detect_people(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def detect_people():
    global person_id_counter
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        for person_info in detected_people.values():
            person_info["alarm_triggered"] = False

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    centroid_x = x + w // 2
                    centroid_y = y + h // 2

                    matched_person_id = None
                    for person_id, person_info in detected_people.items():
                        last_centroid_x, last_centroid_y = person_info["centroid"]
                        distance = np.sqrt(
                            (centroid_x - last_centroid_x) ** 2
                            + (centroid_y - last_centroid_y) ** 2
                        )
                        if distance < 50:
                            matched_person_id = person_id
                            break

                    if matched_person_id is None:
                        detected_people[person_id_counter] = {
                            "id": person_id_counter,
                            "centroid": (centroid_x, centroid_y),
                            "bbox": (x, y, w, h),
                            "time": time.time(),
                            "last_update_time": time.time(),
                            "alarm_triggered": False,
                        }
                        person_id_counter += 1
                    else:
                        detected_people[matched_person_id] = {
                            "id": matched_person_id,
                            "centroid": (centroid_x, centroid_y),
                            "bbox": (x, y, w, h),
                            "time": detected_people[matched_person_id]["time"],
                            "last_update_time": time.time(),
                            "alarm_triggered": False,
                        }

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        current_time = time.time()
        for person_info in detected_people.values():
            elapsed_time = current_time - person_info["time"]
            person_info["time"] = person_info["time"] + (
                current_time - person_info["last_update_time"]
            )
            person_info["last_update_time"] = current_time

        # if elapsed_time > setTime and not person_info.get("alarm_triggered", False):
        #  person_info["alarm_triggered"] = True
        #  winsound.Beep(1000, 500)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for person_id, person_info in detected_people.items():
            x, y, w, h = person_info["bbox"]
            elapsed_time = current_time - person_info["time"]
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            waiting_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'Person {person_info["id"]} - Waiting Time: {waiting_time_str}',
                (x, y - 10),
                font,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/waiting_times")
def waiting_times():
    return jsonify(get_waiting_times())


if __name__ == "__main__":
    socketio.run(app, debug=True)
