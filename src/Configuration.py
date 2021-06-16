import logging
def get_logger():
    logger=logging.getLogger()
    handler=logging.StreamHandler()
    formatter=logging.Formatter('%(asctime)s %(levelname)s \t%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
class ConfigurationsPOJO:
    camNo1 = 'cam1'
    camNo2 = 'cam2'
    camNo3 = 'cam3'
    camNo4 = 'cam4'
    TimeFormat='%H:%M:%S'
    timezone='Asia/Chittagong'

    yoloV3CfgPath = "detector/YOLOv3/cfg/yolo_v3.cfg"
    yoloV3WghtsPath = "trackerModels/yolov3.weights"
    yoloV3CocoNamesPath = "detector/YOLOv3/cfg/coco.names"

    url = "mongodb://localhost:27017/"
    dbName = "userDetails"
    collectionName = "userAttendance"
    collectionNameForUnknwnUser = "userAttendanceForUnknwnUsr"

    clssfr_ModelPath = 'faceEmbeddingModels/my_model.h5'