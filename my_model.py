#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 01:08:01 2020

@author: rahul
"""
import cv2
import numpy as np
from pygame import mixer 

# for alarm warning sound
mixer.init()
sound = mixer.Sound("/home/rahul/Social Distancing Detection/alarm.wav")

#Method to calculate calibrated distance 
def calculate_dist(a,b):
     return ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
 
#Function to check the closeness between two point
def isClose(a,b):
    dist = calculate_dist(a,b)
    calib = (a[1] + b[1]) / 2
    if 0 < dist < 0.15 * calib:
        return 1
    else:
        return 0
    
# Loading all class objects and split the names on the basis of new line in coco names file.
objectName_path="/home/rahul/Social Distancing Detection/coco.names"
with open("coco.names", "r") as f: 
  objectName = [line.strip() for line in f.readlines()] 


configFile_path="/home/rahul/Social Distancing Detection/yolov3-tiny.cfg"
weightFile_path="/home/rahul/Social Distancing Detection/yolov3-tiny.weights"

# Loading yolo network using dnn function. It is used to load our model from disk.
yoloNetwork=cv2.dnn.readNetFromDarknet(configFile_path,weightFile_path)


# function to get the output layer names
def get_output_layers(net):
    layer_name = net.getLayerNames()

    output_layers = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


#Object for input video
cap = cv2.VideoCapture("/home/rahul/Social Distancing Detection/input_video.mp4")

#Object for output video
def video_output(frame):
    out = cv2.VideoWriter("/home/rahul/Social Distancing Detection/output_video.mp4", cv2.VideoWriter_fourcc(*"XVID"),15.,(frame.shape[1], frame.shape[0]), isColor=True)
    out.write(frame)
    cv2.imshow("preview", frame)
    
#initialization
HEIGHT=None
WIDTH=None
RedColor=(0, 0, 150)
BlueColor=(225,0,0)

while(True):
    (ret,frame)=cap.read()
    if not ret:
        break
    if WIDTH is None or HEIGHT is None:
        HEIGHT = frame.shape[0]
        WIDTH=frame.shape[1]
        r=WIDTH

    frame = frame[0:HEIGHT, 200:r]
    HEIGHT = frame.shape[0]
    WIDTH=frame.shape[1]
    
    #performing mean subtraction,normalization and swapping of channels on input frame image
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416),swapRB=True, crop=False)
    
    yoloNetwork.setInput(blob)
    layerOutputs = yoloNetwork.forward(get_output_layers(yoloNetwork))
    
    #initialization
    bounding_boxes = []
    confidence_scores = []
    class_ids = []
    threshold_confidence=0.5
    threshold_nms=0.3
    
    # for each detetion from each output layer get the confidence, class id,bounding box dims,
    # and ignoring confidence_scores<0.5
    for op in layerOutputs:
        for detection in op:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if objectName[class_id] == "person":
                if confidence > 0.5:
                    centerX=int(detection[0]*WIDTH)
                    centerY=int(detection[1]*HEIGHT)
                    w=int(detection[2]*WIDTH)
                    h=int(detection[3]*HEIGHT)
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    bounding_boxes.append([x, y, int(w), int(h)])
                    confidence_scores.append(float(confidence))
                    class_ids.append(class_id)
                    
    #Applying Non-Max supression 
    box_NMS = cv2.dnn.NMSBoxes(bounding_boxes, confidence_scores, threshold_confidence, threshold_nms)
    
    if len(box_NMS) > 0:
        
        # a boolean list to store the status of each bounding box whether risk free or in risk
        closeness = []
        index = box_NMS.flatten()
        distance = []
        
        #stores the center of each bounding box remaining after applying NMS
        center = []
        
        for i in index:
            box=bounding_boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            center.append([int(x + w / 2), int(y + h / 2)])
            closeness.append(0)
        
        #check distance for each of the bounding obtained 
        for i in range(len(center)):
            for j in range(len(center)):
                m = isClose(center[i], center[j])
                if m == 1:
                    distance.append([center[i], center[j]])
                    closeness[i] = 1
                    closeness[j] = 1
                    
        # for all the bounding boxes which are identified as close will be highlighted and a warning alarm will play
        idx = 0
        for i in index:
            x = box[0]
            y = box[1]
            start_point=(x,y)
            
            w = box[2]
            h = box[3]
            end_point=(x+w,y+h)
            
            if closeness[idx] == 1:
                try:
                    sound.play()
                except:  # isplaying = False
                    pass
              
                cv2.rectangle(frame, start_point,end_point, RedColor,thickness=2)
            else:
                cv2.rectangle(frame, start_point, end_point, BlueColor, thickness=2)
            idx += 1
        for x in distance:
            cv2.line(frame, tuple(x[0]), tuple(x[1]), RedColor, 2)
            
    # Write the output
    video_output(frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("p"):  # Exit condition
        break
    
    
# When everything done, release the capture
cap.release()

# finally, close window
cv2.destroyAllWindows()
