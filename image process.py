
# importing Libraries
import cv2                  
import os
root="C:/Users/tarun/Downloads"         # innitializing the Image path
dir_name = "happy_2"                # initializing or naming the folder to save out dataset

padding=30      # pading

# Load the pre-trained face detection model from OpenCV and the Cascade Calssifier that will detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')     

# Making a counter to raname the images 
try:
    counter = len(os.listdir(f"extra_data/{dir_name}"))
except:
    os.mkdir(f"extra_data/{dir_name}")
    counter = 0

for img_file in (os.listdir(os.path.join(root, dir_name))):
    
    input_image = cv2.imread(f"{root}/{dir_name}/{img_file}")    # Load the input image
    print(f"{root}/{dir_name}/{img_file}")
    
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)   # Convert the image to grayscale (required for face detection)

    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))        # Perform face detection
    
    for (x, y, w, h) in faces:      # Iterate over detected faces
       
        croped_img=gray_image[(y-padding):(y+padding + h) , (x-padding):(x+padding + w)]        # Draw a rectangle around the detected face

    try:
        
        save_file = f"extra_data/{dir_name}/{dir_name}-{counter}.jpg"       # Save the image with detected faces

        cv2.imwrite(save_file, croped_img)
        counter += 1
    except:
        print("Esceped")
    print(f"Saved {save_file}")
    




