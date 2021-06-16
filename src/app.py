import argparse
import tkinter as tk
from tkinter import *
import logging
from utils_getting_faces.get_faces_from_camera import TrainingDataCollector
from utils_face_embedding.faces_embedding import GenerateFaceEmbedding
from utils_train.train_softmax import TrainFaceRecogModel
from utils_prediction.facePredictor import FacePredictor
class RegistrationModule:
    def __init__(self,logFileName):
        self.logFileName= logFileName
        self.window=tk.Tk()
        self.window.title('Face Recognition and Tracking')

        self.window.resizable(0,0)
        window_height=600
        window_width=880

        screen_width=self.window.winfo_screenwidth()
        screen_height=self.window.winfo_screenheight()
        x_cordinate=int((screen_width/2)-(window_width/2))
        y_cordinate=int((screen_height/2)-(window_height/2))
        self.window.geometry('{}x{}+{}+{}'.format(window_width,window_height,x_cordinate,y_cordinate))
        self.window.configure(bg='#ffffff')

        self.window.grid_rowconfigure(0,weight=1)
        self.window.grid_columnconfigure(0,weight=1)

        header=tk.Label(self.window,text='Employee Monitoring Registration',width=80,height=2,fg='white',bg='#363e75',
                        font=('time',18,'bold','underline'))
        header.place(x=0,y=0)
        clientID=tk.Label(self.window,text='Client ID',width=10,height=2,fg='white',bg='#363d75',font=('times',15))
        clientID.place(x=80,y=80)
        displayVariable=StringVar()
        self.clientIDTxt=tk.Entry(self.window,width=20,text=displayVariable,bg='white',fg='black',
                                 font=('times',15,'bold'))
        self.clientIDTxt.place(x=205,y=80)
        empID = tk.Label(self.window, text="EmpID", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empID.place(x=450, y=80)

        self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empIDTxt.place(x=575, y=80)

        empName = tk.Label(self.window, text="Emp Name", width=10, fg="white", bg="#363e75", height=2,
                           font=('times', 15))
        empName.place(x=80, y=140)

        self.empNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empNameTxt.place(x=205, y=140)

        emailId = tk.Label(self.window, text="Email ID :", width=10, fg="white", bg="#363e75", height=2,
                           font=('times', 15))
        emailId.place(x=450, y=140)
        self.emailIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.emailIDTxt.place(x=575, y=140)

        mobileNo = tk.Label(self.window, text=" Mobile No:", width=10, fg="white", bg="#363e75", height=2,
                            font=('times', 15))
        mobileNo.place(x=450, y=140)

        self.mobileNoTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.mobileNoTxt.place(x=575, y=140)
        lbl3=tk.Label(self.window,text='Notification:',width=15,fg='white',bg='#363e75',height=2,
                      font=('time',15))
        lbl3.place(x=80, y=260)
        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2,
                                activebackground="#bbc7d4",
                                font=('times', 15))
        self.message.place(x=205, y=260)
        takeImg=tk.Button(self.window,text='Take Images',command=self.collectUserImageForRegistration,fg='white',
                          bg='#363e75',height=2,activebackground='#118ce1',font=('times',15,'bold'))
        takeImg.place(x=80,y=350)
        trainImg=tk.Button(self.window,text='Train Images',command=self.trainModel,fg='white',bg='#363e75',
                           height=2,activebackground='#118ce1',font=('times',15,'bold'))
        trainImg.place(x=350,y=350)
        predictImg=tk.Button(self.window,text='Predict',command=self.make_predicition,fg='white',bg='#363e75',
                             width=15,height=2,activebackground='#118ce1',font=('times',15,'bold'))
        predictImg.place(x=600,y=350)

        quitWindow=tk.Button(self.window,text="Quit",command=self.close_window,fg='white',bg='#363e75',width=10,height=2,
                             activebackground='#118ce1',font=('times',15,'bold'))
        quitWindow.place(x=650,y=510)

        #label=tk.Label(self.window)

        self.window.mainloop()
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',filename=self.logFileName,
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


    def collectUserImageForRegistration(self):
        # clientIDVal = (self.clientIDTxt.get())
        # empIDVal = self.empIDTxt.get()
        name = (self.empNameTxt.get())
        ap=argparse.ArgumentParser()

        ap.add_argument('--faces',default=50,help='Number of faces that camera will get')
        ap.add_argument('--output',default='../dataset/train/'+name,
                        help='Path to faces output')
        args=vars(ap.parse_args())
        trnngDataCollctrObj=TrainingDataCollector(args)
        trnngDataCollctrObj.collectImagesFromCamera()
        notifctn='We have collected'+str(args['faces'])+ ' images for training'
        self.message.configure(text=notifctn)
    def getFaceEmbedding(self):
        ap=argparse.ArgumentParser()
        ap.add_argument('--dataset',default='../dataset/train',
                        help='Path to training dataset')
        ap.add_argument('--embeddings',default='faceEmbeddingModels/embeddings.pickle')
        ap.add_argument('--image-size',default='112,112',help='')
        ap.add_argument('--model',default='../insightface/models/model-y1-test2/model,0',help='path to load model')
        ap.add_argument('--ga-model',default='',help='path to load model')
        ap.add_argument('--gpu',default=0,type=int,help='gpu id')
        ap.add_argument('--det',default=0,type=int,
                        help='mtcnn option, 1 means using R+0, 0 means detect from begining')
        ap.add_argument('--flip',default=0,type=int,help='whether dolr fllip aug')
        ap.add_argument('--threshold',default=1.24,type=float,help='ver dist threshold')
        args=ap.parse_args()
        genFaceEmbdng=GenerateFaceEmbedding(args)
        genFaceEmbdng.genFaceEmbedding()
    def trainModel(self):
        ap=argparse.ArgumentParser()
        ap.add_argument('--embeddings',default='faceEmbeddingModels/embeddings.pickle',
                        help='path to serialized db of facial embeddings')
        ap.add_argument('--model',default='faceEmbeddingModels/my_model.h5',help='path to output trained model')
        ap.add_argument('--le',default='faceEmbeddingModels/le.pickle',help='path to output label encoder')
        args=vars(ap.parse_args())
        self.getFaceEmbedding()
        faceRecogModel=TrainFaceRecogModel(args)
        faceRecogModel.trainKerasModelForFaceRecognition()

        notifctn='Model training is successful.Now you can go for prediction'
        self.message.configure(text=notifctn)
    def make_predicition(self):
        faceDetector=FacePredictor()
        faceDetector.detectFace()
    def close_window(self):
        self.window.destroy()




logFileName='riad.txt'
regmodule=RegistrationModule(logFileName)


