from ultralytics import *
from sort import *
from backsubstraction import *
import cv2
import matplotlib.path as mplPath
import cvzone
import math

type = "web"

def Processor(filename, modelt, aoi):

    if modelt == "YOLOv8n":
        model = YOLO('../Yolo-Weights/bestn.pt')
        print("hello")
    elif modelt == "YOLOv8l":
        model = YOLO('../Yolo-Weights/best.pt')
    elif modelt == "Image Processing":
        objectdetector = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=200)

    yoloclassNames = ["bus", "car", "motorbike", "truck"]

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    totalCount = []

    if(type == "web"):
        video = cv2.VideoCapture(f'uploads/{filename}')
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    else:
        video = cv2.VideoCapture(0) 
    fid = 0
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    mask = np.zeros((frame_width, frame_height, 3))

    pointlst = []
    for coord in aoi:
        x = coord['x']
        y = coord['y']
        rti = frame_height / 450
        yt = int(y) * rti
        diff = (800 * rti - frame_width) / 2
        xt = int(x) * rti - diff
        pointlst.append([int(xt), int(yt)])

    pointsnp = np.array(pointlst)
    poly_path = mplPath.Path(pointsnp)

    fresult = cv2.VideoWriter('output.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              10, size)
    while True:

        ret, frame = video.read()
        cv2.polylines(frame, pts=[pointsnp], isClosed=True, color=(0, 255, 0), thickness=2)
        fid += 1

        if ret == True:
            if modelt == "YOLOv8n" or modelt == "YOLOv8l":
                pred = model(frame, stream=True)
                print("running")
                detections = np.empty((0, 5))
                for r in pred:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1

                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentClass = yoloclassNames[cls]

                        if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.2:
                            if poly_path.contains_point((cx, cy)):
                                currentArray = np.array([x1, y1, x2, y2, conf])
                                detections = np.vstack((detections, currentArray))

                resultsTracker = tracker.update(detections)

                for result in resultsTracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w = x2 - x1
                    h = y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                    if id not in totalCount:
                        totalCount.append(id)

            elif (modelt == "Image Processing" and fid>1):
                detections = np.empty((0, 5))
                #mask = objectdetector.apply(frame)
                mask = objdet(frame,grayprev,frame_height,frame_width)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x1, y1, w, h = cv2.boundingRect(cnt)
                    x2 = x1 + w
                    y2 = y1 + h
                    cx = x1 + (w / 2 )
                    cy = y1 + (h / 2)
                    if area > 100 and area < 20000: #w > 50 and h > 50 and w < 300 and h < 300:
                        if poly_path.contains_point((cx, cy)):
                                currentArray = np.array([x1, y1, x2, y2, 0.8])
                                detections = np.vstack((detections, currentArray))
                
                resultsTracker = tracker.update(detections)

                for result in resultsTracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w = x2 - x1
                    h = y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                    if id not in totalCount:
                        totalCount.append(id)
            if(type=="web"):
                progress = (fid / length) * 100
            cv2.putText(frame, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
            fresult.write(frame)
            grayprev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return len(totalCount)
        

    video.release()
    fresult.release()
    return -1
