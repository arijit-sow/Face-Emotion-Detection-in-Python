import cv2 as cv
import csv
import os
import tqdm
import numpy as np
picture_size = 48
folder_path = "E:/git file/Emotion_Detection_CNN/data/train"
# train/

index = 0
for dir in os.listdir(folder_path):
    print(dir)
    i = 0
    for img in os.listdir(os.path.join(folder_path, dir)):
        # print(f"{folder_path}/{dir}/{img}")
        img = cv.imread(f"{folder_path}/{dir}/{img}")
        # img = tqdm()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (picture_size, picture_size))
        img = np.asarray(img)
        
        flatten_img = img.flatten()
        normalized_img_array = [(npx/255) for npx in flatten_img]
        
        with open("dataset_X.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([*flatten_img])
            # break

        with open("dataset_Y.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([index])
            # break

        i+=1
    index+=1
    
