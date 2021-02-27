#import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import csv
import tensorflow as tf
import shutil  
import random

FP = []
FN = []
total_neg_count = 0
total_pos_count = 0

#Argument parser
ap=argparse.ArgumentParser()
ap.add_argument('-n','--image1',type=str,default='',help='path')
ap.add_argument('-p','--image2',type=str,default='',help='path')
ap.add_argument('-m','--model',type=str,default='',help='path')
ap.add_argument('-dn','--destination1', type=str, default='true', help="path to destination folder")
ap.add_argument('-dp','--destination2', type=str, default='true', help="path to destination folder")
args=vars(ap.parse_args())

FP_dest = args['destination1']
FN_dest = args['destination2']
negative_folder= args['image1']
positive_folder = args['image2']
heart= args['model']

#Make directory's for moving images
os.makedirs(FP_dest,exist_ok=True)
os.makedirs(FN_dest,exist_ok=True)

#Preprocessing the image
def preprocessing_img(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(224, 224),interpolation = "lanczos")    
    img_array = tf.keras.preprocessing.image.img_to_array(img)    
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array
   
model = tf.keras.models.load_model(heart) 
negative_folder_path = os.listdir(negative_folder)
positive_folder_path = os.listdir(positive_folder)

for each_negative_image in negative_folder_path:
    total_neg_count += 1 
    try:
        img_path = os.path.join(negative_folder,each_negative_image)
        new_image = preprocessing_img(img_path)
        pred = model.predict(new_image)
        print(each_negative_image , pred)
        if pred[0][0]>=0.7:
            FP.append(each_negative_image)
        else:
            continue        
    except:
        print(each_negative_image)
        continue
for each_positive_image in positive_folder_path:
    total_pos_count += 1 
    try:
        img_path = os.path.join(positive_folder,each_positive_image)
        new_image = preprocessing_img(img_path)
        pred = model.predict(new_image)
        print(each_positive_image, pred)
        if pred[0][0]<0.7:
            FN.append(each_positive_image)            
        else:
            continue        
    except:
        print(each_positive_image)
        continue
for x_call in FP:
    shutil.move(os.path.join(negative_folder,x_call),FP_dest)
for x_call in FN:
    shutil.move(os.path.join(positive_folder,x_call),FN_dest)

Negative_accuracy = (((total_neg_count - len(FP))*100)/(total_neg_count))
Positive_accuracy = (((total_pos_count - len(FN))*100)/(total_pos_count))
print('Negative count:',total_neg_count)
print('Positive count:',total_pos_count)
print('FP count:',len(FP))
print('FN count:',len(FN))
print('Positive Accuracy:',Positive_accuracy)
print('Negative Accuracy:',Negative_accuracy)
