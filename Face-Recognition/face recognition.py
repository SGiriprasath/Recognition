import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

Tk().withdraw()

load_image = askopenfilename()
target_image = askopenfilename()
target_encoding = fr.face_encodings(fr.load_image_file(target_image))[0]

def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_image = fr.load_image_file(f'{folder}/{filename}')
        known_encoding = fr.face_encodings(known_image)[0]
        list_people_encoding.append((known_encoding, filename))

    return list_people_encoding

def find_target_face():
    face_locations = fr.face_locations(fr.load_image_file(target_image))

    for person in encode_faces('people'):
        encoded_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces([encoded_face], target_encoding, tolerance=0.55)[0]
        print(f'{filename} is {"Giri" if is_target_face else "Not Giri"}')

        if any(face_locations):
            face_number = 0
            for location in face_locations:
                if is_target_face:
                    label = "Giri"
                    create_frame(location, label)
                else:
                    label = "Not Giri"
                    create_frame(location, label)
                face_number += 1

def create_frame(location, label):
    top, right, bottom, left = location

    # Load the target image separately for drawing rectangles
    draw_image = cv.imread(target_image)
    cv.rectangle(draw_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(draw_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(draw_image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

    # Display the image with rectangles
    render_image(draw_image)

def render_image(image):
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition', rgb_img)
    cv.waitKey(0)

find_target_face()
