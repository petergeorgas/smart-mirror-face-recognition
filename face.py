import time
import requests
import os
import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)

video_capture = cv2.VideoCapture(0)


face_files = os.listdir("faces")

known_face_encodings = []
known_face_names = []

# Load in our faces
for file in face_files:
    file_name = os.path.splitext(file)[0]

    img = face_recognition.load_image_file(os.path.join("faces", file))
    img_face_enc = face_recognition.face_encodings(img)[0]

    known_face_names.append(file_name)
    known_face_encodings.append(img_face_enc)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

name_map = {
    "Peter Georgas": {"name": "Peter", "id": "RQQo7YZKGeT1EHgZPP1mMg2twJh2"},
    "Wild Bill": {"name": "William", "id": "GAnmE0SGijVVPkV2a5gZyZm65753"},
    "Jonah Eck": {"name": "Jonah", "id": "S5OezfUBMnWknbVyEP3PpqPPkMF3"},
    "Brad Hetrick": {"name": "Brad", "id": "Z8BhwkLaJdXUPeQjWBc1XDIngYp1"},
    "Logan Rising": {"name": "Logan", "id": "GHDFA3g1ngVhqCZK1L0lFaMeL8K2"},
}

last_seen = None

interval_start_time = time.time()

print("FACE FINDER STARTED...")
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        face_found = False
        # Resize frame of video to 1/5 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        small_frame = cv2.rotate(small_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        """
        if len(face_locations) == 0 and time_elapsed > 15:
            last_seen = None
            interval_start_time = time.time()  # Reset the time interval
            continue
        """

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []

        if len(face_encodings) == 0:  # No faces found
            face_found = False
            now = time.time()
            time_elapsed = now - interval_start_time
            if time_elapsed > 15:
                print("sending reset")
                reset = {"name": "reset", "id": "reset"}
                try:
                    requests.post("http://localhost:8080/face", json=reset)
                except Exception:
                    print("[ERROR] Failed to send reset signal to api gateway.")
                last_seen = None
                interval_start_time = time.time()

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                if name != last_seen:
                    last_seen = name
                    print(f"{name} detected. Sending request.")
                    try:
                        requests.post("http://localhost:8080/face", json=name_map[name])
                        face_found = True
                    except Exception:
                        print("[ERROR] Failed to send reset signal to api gateway.")

                    break
                else:
                    face_found = True
            else:  # No match found
                now = time.time()
                time_elapsed = now - interval_start_time
                if time_elapsed > 15:
                    print("sending reset")
                    reset = {"name": "reset", "id": "reset"}
                    try:
                        requests.post("http://localhost:8080/face", json=reset)
                    except Exception as e:
                        print("[ERROR] Failed to send face information to api gateway.")
                    last_seen = None
                    interval_start_time = time.time()

            face_names.append(name)

            """
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name != last_seen:
                    last_seen = name
                    print(f"i see {name}")
                    print("sending request!")
                    # requests.post("http://localhost:8080/face", json=name_map[name])
                    face_found = True
                    break
                else:
                    face_found = True

            face_names.append(name)
            """

    process_this_frame = not process_this_frame
    if face_found:
        print("timer started")
        time.sleep(15)
        print("timer expired")
        face_found = False

    # This will not need to be displayed in actual implementation

    """
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        ## Scale back up face locations since the frame we detected in was scaled to 1/5 size
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    """


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
