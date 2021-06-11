from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
model=MTCNN()
filename='images.jpeg'
pixels=plt.imread(filename)
def draw_image_with_boxes(filename,result_list):
    data=plt.imread(filename)
    plt.imshow(data)
    ax=plt.gca()
    for result in result_list:
        x,y,width,height=result['box']
        rect=Rectangle((x,y),width,height,color='red',fill=False)
        ax.add_patch(rect)
        for key,value in result['keypoints'].items():
            dot=Circle(value,radius=2,color='red')
            ax.add_patch(dot)
    plt.show()
def draw_faces(filename,result_list):
    data=plt.imread(filename)
    for i in range(len(result_list)):
        x1,y1,width,height=result_list[i]['box']
        x2,y2=x1+width,y1+height
        plt.subplot(1,len(result_list),i+1)
        plt.axis('off')
        plt.imshow(data[y1:y2,x1:x2])
    plt.show()

faces=model.detect_faces(pixels)
#draw_image_with_boxes(filename,faces)
draw_faces(filename,faces)
#for face in faces:
   # print(face)