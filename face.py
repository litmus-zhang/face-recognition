import face_recognition
import cv2
import numpy as np
import pyttsx3
# pip install pyttsx4
import pyttsx4
import speech_recognition as sr
import asyncio
import datetime
import time
import threading


#engine = pyttsx3.init(driverName="espeak")
engine = pyttsx4.init('coqui_ai_tts')
engine.setProperty('speaker_wav', './main.wav')
voices = engine.getProperty("voices")
engine.setProperty("rate", 10)
engine.setProperty("voice", voices[11].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    

def wishMe(name: str = ""):
    hour = int(datetime.datetime.now().hour)
    time.sleep(3)
    if hour >= 0 and hour < 12:
        speak("Good Morning " + name)

    elif hour >= 12 and hour < 18:
        speak("Good Afternoon " + name)

    else:
        speak("Good Evening " + name)

    assname = ("Okikiola 1 point o")
    speak("I am Okikiola, your Assistant")
    speak(assname)


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other
# demos that don't require it instead.
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(2)

# Load  pictures and learn how to recognize it.
dr_oloodo = face_recognition.load_image_file("./training/dr-oloodo/img_1.jpeg")
mr_andrew = face_recognition.load_image_file("./training/andrew/img_1.jpeg")
mrs_aisha = face_recognition.load_image_file("./training/aisha/img_1.jpeg")
dr_oloodo_face_encoding = face_recognition.face_encodings(dr_oloodo)[0]
mr_andrew_face_encoding = face_recognition.face_encodings(mr_andrew)[0]
mrs_aisha_face_encoding = face_recognition.face_encodings(mrs_aisha)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    dr_oloodo_face_encoding,
    mr_andrew_face_encoding,
    mrs_aisha_face_encoding
]
known_face_names = [
    "Dr Oloodo",
    "Mr Andrew David",
    "Mrs Abdulkadir Binta"
]


def main():

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition
            # processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of
            # video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Person"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the
                # new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name == "Person":
                        x = threading.Thread(target=wishMe, args=("",))
                        x.start()

                    else:
                        x = threading.Thread(target=wishMe, args=(name,))
                        x.start()
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (
                top, right, bottom, left), name in zip(
                face_locations, face_names):
            # Scale back up face locations since the frame we detected in was
            # scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


main()
