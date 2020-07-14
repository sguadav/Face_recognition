import cv2
import face_recognition as fr
from datetime import datetime


def findEncoding(list_img):
    encode_list = []
    for img_l in list_img:
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        encoded = fr.face_encodings(img_l)[0]
        encode_list.append(encoded)
    return encode_list


def markAttendance(name):
    with open('attendance_tracker.csv', 'r+') as f:
        data_list = f.readlines()
        name_list_csv = []
        for line in data_list:
            entry = line.split(',')
            name_list_csv.append(entry[0])
        if name not in name_list_csv:
            now = datetime.now().strftime('%H:%M:%S')
            f.writelines(f'\n{name},{now}')