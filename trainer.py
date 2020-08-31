import pickle
import os,cv2
import numpy as np
from PIL import Image;
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]
    faceSamples=[]
    count_id=[]
    name_and_ids={}
    for name in dirlist:
        Ids=[]
        imagePaths=[os.path.join(path+name+"/",f) for f in os.listdir(path+name+"/")]
        for imagePath in imagePaths:
        	#loading the image and converting it to gray scale
        	pilImage=Image.open(imagePath).convert('L')
        	#Now we are converting the PIL image into numpy array
        	imageNp=np.array(pilImage,'uint8')
        	#getting the Id from the image
        	Id=int(os.path.split(imagePath)[-1].split(".")[1])
        	# extract the face from the training image sample
        	faces=detector.detectMultiScale(imageNp)
        	for (x,y,w,h) in faces:
            		faceSamples.append(imageNp[y:y+h,x:x+w])
            		count_id.append(Id)
            		Ids.append(Id)
        name_and_ids.update({name:Ids})
    return faceSamples,count_id,name_and_ids
faces,Ids,name = getImagesAndLabels('images/')
print(name)
print(Ids)
with open("labels","wb" ) as f:
 	pickle.dump(name,f)
recognizer.train(faces,np.array(Ids))
print("Successfully trained")
recognizer.save('trainer.yaml')
