import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import KFold
from src.configuration import get_logger
from src.utils_train.softmax import Softmax

class TrainFaceRecogModel:
    def __init__(self,args):
        self.args=args
        self.logger=get_logger()
        self.data=pickle.loads(open(args['embeddings'],'rb').read())
    def trainKerasModelForFaceRecognition(self):
        le=LabelEncoder()
        labels=le.fit_transform(self.data['names'])
        num_classes=len(np.unique(labels))
        labels=labels.reshape(-1,1)
        one_hot_encoder=OneHotEncoder(categorical_features=[0])
        labels=one_hot_encoder.fit_transform(labels).toarray()
        embeddings=np.array(self.data['embeddings'])

        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]
        softmax=Softmax(input_shape=(input_shape,),num_classes=num_classes)
        model=softmax.build()

        cv=KFold(n_splits=5,random_state=42,shuffle=True)
        history={'acc': [], 'val_acc': [],'loss': [],'val_loss': []}

        for train_idx,valid_idx in cv.split(embeddings):
            X_train,X_val,y_train,y_val=embeddings[train_idx],embeddings[valid_idx],labels[train_idx],labels[valid_idx]
            his=model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,validation_data=(X_val,y_val))
            print(his.history['acc'])

            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']
            self.logger.info(his.history['acc'])
        model.save(self.args['model'])
        f=open(self.args['le'],'wb')
        f.write(pickle.dumps(le))
        f.close()
