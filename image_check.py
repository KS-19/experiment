from PIL import Image
import numpy as np
import time

image=np.loadtxt("gray_image.csv",delimiter=",")
posture = np.loadtxt("posture_0710_2.csv",delimiter=",")
posture = np.round(posture)

label=np.array([])

for i in range(len(posture)):
    if(posture[i]>=0 and posture[i]<72):
        label = np.append(label,posture[i])
    elif(posture[i]>=72 and posture[i]<144):
        label = np.append(label,posture[i]-72)
    elif(posture[i]>=144 and posture[i]<216):
        label = np.append(label,posture[i]-144)
    elif(posture[i]>=216 and posture[i]<288):
        label = np.append(label,posture[i]-216)
    else:
        label = np.append(label,posture[i]-288)
posture=label
        
def show(num):
    list=np.array([])
    for i in range(len(posture)):
        if posture[i]==num:
            list = np.append(list, i)
    print (list)
    for j in range(len(list)):
        pilImg = Image.fromarray(np.uint8(np.reshape(image[int(list[j])],(64,64))))
        pilImg.show()

