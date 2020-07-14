import cv2
import numpy as np
import face_recognition as fr
import os
from helper_func import findEncoding
from helper_func import markAttendance

# import images
path = "images_basics/train"
img_list_train = []
names_list = []
my_images_file_list = os.listdir(path)
print(my_images_file_list)

for cls in my_images_file_list:
    cur_img = cv2.imread(f'{path}/{cls}')
    img_list_train.append(cur_img)
    names_list.append(os.path.splitext(cls)[0])

encoded_list = findEncoding(img_list_train)
print("Encoding Complete")

# Using the Web cam
video_Cap = cv2.VideoCapture(0)

while True:
    success, img_comparing = video_Cap.read()
    img_comparing_resized = cv2.resize(img_comparing, (0, 0), None, 0.25, 0.25)
    img_comparing_resized = cv2.cvtColor(img_comparing_resized, cv2.COLOR_BGR2RGB)

    face_location_curr_frame = fr.face_locations(img_comparing_resized)
    encode_curr_frame = fr.face_encodings(img_comparing_resized, face_location_curr_frame)

    for encode_face, face_loc in zip(encode_curr_frame, face_location_curr_frame):
        # Comparing the Web cam with the images
        matches = fr.compare_faces(encoded_list, encode_face)
        face_dist = fr.face_distance(encoded_list, encode_face)
        print(face_dist)
        # index in the list
        match_index = np.argmin(face_dist)

        # When the images list and the Webcam match, proceed with the following
        if matches[match_index]:
            name_matched = names_list[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img_comparing, (x1, y1), (x2, y2), (23, 186, 255), 2)
            cv2.rectangle(img_comparing, (x1, y2-35), (x2, y2), (23, 186, 255), cv2.FILLED)
            cv2.putText(img_comparing, name_matched, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)
            markAttendance(name_matched)

    cv2.imshow('Webcam', img_comparing)
    cv2.waitKey(1)
