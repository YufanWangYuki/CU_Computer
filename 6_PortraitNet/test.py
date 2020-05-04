import numpy as np
from PIL import Image
import keras
import sys
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2

# Load the model
model=load_model('deconv_fin_munet.h5')

# Load a test image
im=Image.open('6916.jpg')
print(im.size)
# Inference
im=im.resize((128,128),Image.ANTIALIAS)
img=np.float32(np.array(im)/255.0)
img=img[:,:,0:3]

# Reshape input and threshold output
out=model.predict(img.reshape(1,128,128,3))
out=np.float32((out>0.5)).reshape(128,128,1)

#result = (out*img).reshape((600,800),Image.ANTIALIAS)
#print(result)
# Input image
plt.figure("Input")
plt.imshow(img)


# Output image
plt.figure("Output")
plt.imshow(out*img)

print(type(result))

#result_img = Image.fromarray(result)
cv2.imwrite("Portrait_BG.jpg", result) 

plt.show()

#Sample run: python test.py test/four.jpeg
