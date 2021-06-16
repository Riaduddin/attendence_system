from imutils import paths
import numpy as np
import pickle
import cv2
from src.insightface.deploy import face_model
import os
class GenerateFaceEmbedding:
    def __init__(self,args):
        self.args=args
        self.image_size='112,112'
        self.model='./insightface/models/model-y1-test2/model,0'
        self.threshold=1.24
        self.det=0
    def genFaceEmbedding(self):
        print('[INFO] quantifying faces...')
        imagePaths=list(paths.list_images(self.args.dataset))

        embedding_model=face_model.FaceModel(self.image_size,self.model,self.threshold,self.det)
        knownEmbeddings=[]
        knownNames=[]
        total=0
        for (i,imagePath) in enumerate(imagePaths):
            print('[INFO] processing image {}/{}'.format(i+1,len(imagePaths)))
            name=imagePath.split(os.path.sep)[-2]
            image=cv2.imread(imagePath)
            nimg=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            nimg=np.transpose(nimg,(2,0,1))
            face_embedding=embedding_model.get_feature(nimg)
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total+=1
        print(total,'faces embedded')

        data={'embeddings': knownEmbeddings,'names':knownNames}
        f=open(self.args.embeddings,'wb')
        f.write(pickle.dumps(data))
        f.close()


