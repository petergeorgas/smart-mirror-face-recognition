import time
import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
peter_image = face_recognition.load_image_file("faces/Peter Georgas.jpg")
peter_face_enc = face_recognition.face_encodings(peter_image)[0]

# Load a second sample picture and learn how to recognize it.
bill_image = face_recognition.load_image_file("faces/Wild Bill.jpg")
bill_face_enc = face_recognition.face_encodings(bill_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [peter_face_enc, bill_face_enc]
known_face_names = ["Peter Georgas", "Wild Bill"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

last_seen = set()

interval_start_time = time.time()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/5 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        curr_time = time.time()

        time_elapsed = curr_time - interval_start_time

        if len(face_locations) == 0 and time_elapsed > 60:
            last_seen.clear()
            continue

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )

            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name not in last_seen:
                    last_seen.add(name)
                    print("sending request!")

            face_names.append(name)

    process_this_frame = not process_this_frame

    # This will not need to be displayed in actual implementation

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


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()