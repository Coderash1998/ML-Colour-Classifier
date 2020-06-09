#Code for creating the dataset

from PIL import Image, ImageDraw
import cv2
import matplotlib.image as mpimg 
import os
from matplotlib.pyplot import imshow
r=0
g=0
b=0
path_red="C:/Users/Coderash/Desktop/Dataset/Training/Red/"
os.mkdir(path_red)
path_green="C:/Users/Coderash/Desktop/Dataset/Training/Green/"
os.mkdir(path_green)
path_blue="C:/Users/Coderash/Desktop/Dataset/Training/Blue/"
os.mkdir(path_blue)
for i in range(0,256,10):
    for j in range(0,256,10):
        for k in range(0,256,10):
            img=Image.new('RGB', (64,64), color=(i,j,k))
            if(i>j and i>k):
                img.save(path_red+'Red'+str(r)+'.png')
                r+=1
            elif(j>i and j>k):
                img.save(path_green+'Green'+str(g)+'.png')
                g+=1
            elif(k>i and k>j):
                img.save(path_blue+'Blue'+str(b)+'.png')
                b+=1           
print("Dataset Created")
