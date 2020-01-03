
import cv2
import numpy as np
from PIL import Image
from keras import models

#Load the saved modelcd 
model = models.load_model('Emotion.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 64x64 because we trained the model with this image size.
        im = im.resize((64,64))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        img_array =img_array/255.0
        #Calling the predict method on model to predict the emotion of the person in the image
        prediction = int(model.predict(img_array)[0][0])
        if prediction == 0:
                print("0")
        if prediction==1:
                print("1")
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
