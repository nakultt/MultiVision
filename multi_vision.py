import cv2
from ultralytics import YOLO

model=YOLO("yolo12l.pt")

def detect_obj(vid_src=0):
    vid=cv2.VideoCapture(vid_src)
    if not vid.isOpened():
        print(f"Error: Unable to open video source {vid_src}")
        return
    
    while True:
        success,frame=vid.read()
        if not success:
            print("Error: Unable to read frame")
            break

        results=model(frame)

        for result in results:
            boxes=result.boxes
            for box in boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                confidence=box.conf[0]
                class_id=int(box.cls[0])
                label=model.names[class_id]

                if confidence>0.5:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{label} {confidence:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            
        cv2.imshow("MultiVision",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

detect_obj(0)