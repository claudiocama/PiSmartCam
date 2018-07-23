#!/usr/bin/env python
from flask import Flask, render_template, Response, request, flash
import cv2
import face_recognition
import training
import os, shutil, time

train_model = training.Training()

photo = False

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def get_cam():
    global photo
    global video
    global fourcc
    global out

    camera_port = 0
    camera = cv2.VideoCapture(camera_port)

    while True:
        known_face_encodings = train_model.get_encodings()
        known_face_names = train_model.get_names()
        ret, frame = camera.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, top + 35), (right, top), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, top + 25), font, 1.0, (0, 0, 0), 1)
            if photo:
                cv2.imwrite("test.jpg", frame)
        return frame


def get_frame():
    while True:
        try:
            frame = get_cam()
            frame_encoded = (cv2.imencode('.jpg', frame)[1]).tostring()
        except Exception:
            time.sleep(0.5)
            frame_encoded = b''
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + frame_encoded + b'\r\n')



@app.route('/camera')
def camera():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snap')
def snap():
    global photo
    photo = True
    time.sleep(1)
    photo = False
    return ""


@app.route('/train', methods=['GET', 'POST'])
def train():
    users = os.listdir(train_model.get_directory() + "/")
    if request.method == 'POST':
        if "get_user" in request.form:
            user_selected = request.form["get_user"]
            if 'View' in request.form:
                photos = []
                for photo in os.listdir(train_model.get_directory() + "/" + user_selected):
                    photos.append(train_model.get_directory() + "/" + user_selected + "/" + photo)
                return render_template('train.html', users=users, photos=photos, user_selected=user_selected)
            elif "Del" in request.form:
                shutil.rmtree(train_model.get_directory() + "/" + user_selected)
            elif "Train" in request.form:
                train_model.train()
    if request.args.get("name"):
        if not os.path.exists("static/dataset/"+request.args.get("name")):
            os.mkdir("static/dataset/"+request.args.get("name"))
        else:
            return "The folder already exists"
    users = os.listdir(train_model.get_directory() + "/")
    return render_template('train.html', users=users)


if __name__ == '__main__':
    app.run(host='localhost', debug =True, threaded=True)