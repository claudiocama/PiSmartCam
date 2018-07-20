#!/usr/bin/env python
from flask import Flask, render_template, Response
import cv2
import face_recognition
import training

train_model = training.Training()
train_model.train()
train_model.save()
train_model.load()
known_face_encodings = train_model.get_encodings()
known_face_names = train_model.get_names()


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def get_frame():
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    i = 1
    while True:
        ret, frame = camera.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        frame_encoded = (cv2.imencode('.jpg', frame)[1]).tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + frame_encoded + b'\r\n')
        i += 1

    del (camera)


@app.route('/camera')
def camera():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)