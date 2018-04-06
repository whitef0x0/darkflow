
import math
import cv2
import os
import json

from socketIO_client import SocketIO, LoggingNamespace
import time
import uuid
from time import time as timer

current_milli_time = lambda: int(round(timer() * 1000))

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier("./cv2_data/haar_cascades/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./cv2_data/face_recog/david_helen_model.yaml")

from ...net.detect_sidewalk import run_sidewalk_detection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from google_speech import Speech
import numpy as np

#Default values for google text-to-speech
LANG = "en"
sox_effects = ("speed", "1.5")

#Labels that map names to OpenCV labels
label_to_name = {
    "1": "David Baldwin",
    "2": "Helen Zhang"
}

ds = True
try :
    from deep_sort.application_util import preprocessing as prep
    from deep_sort.application_util import visualization
    from deep_sort.deep_sort.detection import Detection
except :
    ds = False


def expit(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

#Find bounding boxes given a net
def findboxes(self, net_out):
    # meta
    meta = self.meta
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

#Extract boxes into cv shapes
def extract_boxes(self,new_im):
    cont = []
    new_im=new_im.astype(np.uint8)
    ret, thresh=cv2.threshold(new_im, 127, 255, 0)
    p, contours, hierarchy=cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt=contours[i]
        x, y, w, h=cv2.boundingRect(cnt)
        if w*h > 30**2 and ((w < new_im.shape[0] and h <= new_im.shape[1]) or (w <= new_im.shape[0] and h < new_im.shape[1])):
            if self.FLAGS.tracker == "sort":
                cont.append([x, y, x+w, y+h])
            else : cont.append([x, y, w, h])
    return cont

tracked_objects = {}

#Generate Speech Text for the current frame and play them using Google's TTS API 
def object_detection_speech(speech_flag, new_objects, old_objects, height, width):
    object_changes = find_object_changes(old_objects, new_objects)
    return speak_object_changes(speech_flag, object_changes, height, width)

#Find all positional changes for objects between current frame and previous
def find_object_changes(old_objects, new_objects):
    object_changes = {}

    for key, value in new_objects.items():
        output_obj = {
            "bbox": value["bbox"],
            "appeared": False,
            "disappeared": False
        }

        if "person_name" in value:
            output_obj["person_name"] = value["person_name"]

        if key not in old_objects:
            output_obj["appeared"] = True

        object_changes[key] = output_obj 

    for key, value in old_objects.items():
        if key not in new_objects and key not in object_changes:
            object_changes[key] = {
                "bbox": value["bbox"],
                "appeared": False,
                "disappeared": True
            }

            if "person_name" in value:
                object_changes[key]["person_name"] = value["person_name"]

    return object_changes

#Convert object positional change to words
def get_object_position_words(bBox, height, width):
    position_words = ""
    centered = False
    if bBox["bottomright"]["x"] < (width/2):
        #Object is on right side
        position_words += "right "
    elif bBox["topleft"]["x"] > (width/2):
        #Object is on left side
        position_words += "left "
    else:
        centered = True
        position_words += "center "

    if bBox["bottomright"]["y"] < (height/2):
        #Object is on top side
        position_words += "top"
    elif bBox["topleft"]["y"] > (height/2):
        #Object is on bottom side
        position_words += "bottom"
    elif not centered:
        position_words += "center"

    return position_words

speech_out_array = []
old_speech_out_array = []

#Play object change sentences using Google's TTS API
def speak_object_changes(speech_flag, object_changes, height, width):
    global speech_out_array
    global old_speech_out_array
    old_speech_out_array = speech_out_array
    speech_out_array = []

    for key, object_change in object_changes.items():
        speech_out_str = key
        if "person_name" in object_change:
            speech_out_str = object_change["person_name"]

        if object_change["appeared"] is True:
            speech_out_str += " entered frame at "
            speech_out_str += get_object_position_words(object_change["bbox"], height, width)
        elif object_change["disappeared"] is True:
            speech_out_str += " left frame"
        else:
            speech_out_str += " moved to "
            speech_out_str += get_object_position_words(object_change["bbox"], height, width)
        speech_out_array.append(speech_out_str)
    
    output_array = []
    for speech in speech_out_array:

        if speech not in old_speech_out_array:      
            output_array.append(speech)

            if speech_flag is True:
                speech = Speech(speech_out_str, LANG)
                speech.play(sox_effects)
    return output_array

#Detect Faces in an image using OpenCV's faceRecognizer
def detect_face(self, frame):
    """
    detect human faces in image using haar-cascade
    Args:
        frame:
    Returns:
    coordinates of detected faces
    """
    faces = face_cascade.detectMultiScale(frame, 1.1, 2, 0, (20, 20) )
    return faces

#Classify Faces with labels (trained from opencv2_data)
def recognize_face(self, frame_orginal, faces):
    """
    recognize human faces using LBPH features
    Args:
        frame_orginal:
        faces:
    Returns:
        labels and confidence of each predicted person
    """
    predict_label = []
    predict_conf = []
    for x, y, w, h in faces:
        frame_orginal_grayscale = cv2.cvtColor(frame_orginal[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        predict_tuple = recognizer.predict(frame_orginal_grayscale)
        a, b = predict_tuple

        if b > 0.5:
            predict_label.append(a)
            predict_conf.append(b)
    return predict_label, predict_conf

def put_label_on_face(self, frame, faces, labels, confs):
    """
    draw label on faces
    Args:
        frame:
        faces:
        labels:
    Returns:
        processed frame
    """
    i = 0
    for x, y, w, h in faces:
        xA = x
        yA = y
        xB = x + w
        yB = y + h
        if confs[i] < 45:
            actual_label = label_to_name[str(labels[i])] + " " + str(confs[i])
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.putText(frame, actual_label, (x, y), font, 1, (255, 255, 255), 2)
        i += 1
    return frame

def get_label_for_person(self, faces, labels, tracker_bbox):
    """
    @param tracker_bbox:
        formatted like [x1, y1, x2, y2] where (x1, y1) is top left and (x2, y2) is bottom right
    """
    i = 0
    for face_x1, face_y1, face_w, face_h in faces:
        face_x2 = face_x1 + face_w
        face_y2 = face_y1 + face_h

        if face_x1 >= tracker_bbox[0] and face_y1 >= tracker_bbox[1] and face_x2 <= tracker_bbox[2] and face_y2 <= tracker_bbox[3]:
            return label_to_name[str(labels[i])]
        i += 1
    return None

def background_subtraction(self, previous_frame, frame_resized_grayscale, min_area):
    """
    This function returns 1 for the frames in which the area 
    after subtraction with previous frame is greater than minimum area
    defined. 
    Thus expensive computation of human detection face detection 
    and face recognition is not done on all the frames.
    Only the frames undergoing significant amount of change (which is controlled min_area)
    are processed for detection and recognition.
    """
    frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp=0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > min_area:
            temp=1
    return temp     


def postprocess(self,net_out,im,video_id,frame_id = 0,csv_file=None,csv=None,mask = None,encoder=None,tracker=None, previous_frame=None, disable_facial=False):
    """
    Takes net output, draw net_out, save to disk
    """
    start = current_milli_time()
    boxes = self.findboxes(net_out)
    end = current_milli_time()

    time_elapsed = (end - start) / 1000
    #TODO: remove this
    #print("self.findboxes(net_out) took: {}".format(time_elapsed))

    # meta
    meta = self.meta
    nms_max_overlap = 0.1
    threshold = meta["thresh"]
    colors = meta["colors"]
    labels = meta["labels"]
    if type(im) is not np.ndarray:
        im = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape
    thick = int((h + w) // 300)
    resultsForJSON = []


    #Face Detection Code
    label_person_name = ""
    faces = []
    labels = []

    if self.FLAGS.face_recognition and not disable_facial:
        start = current_milli_time()

        # min_area=(3000/800)*im.shape[1] 
        # temp = 1

        frame_grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # if previous_frame is not None:
        #    temp = self.background_subtraction(previous_frame, frame_grayscale, min_area)

        # if temp==1:     
        faces = self.detect_face(frame_grayscale)
        if len(faces) > 0:
            labels, confs = self.recognize_face(imgcv, faces)
            self.put_label_on_face(im, faces, labels, confs)

        end = current_milli_time()
        time_elapsed = (end - start) / 1000
        #TODO: remove this
        #print("face_recognition of single frame took: " + str(time_elapsed))
    speech_actions = None
    if not self.FLAGS.track:
        for b in boxes:
            boxResults = self.process_box(b, h, w, threshold)
            if boxResults is None:
                continue
            left, right, top, bot, label, max_indx, confidence = boxResults
            if self.FLAGS.json:
                resultsForJSON.append({"label": label, "confidence": float("%.2f" % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
                continue
            if self.FLAGS.display:
                cv2.rectangle(imgcv,
                    (left, top), (right, bot),
                    colors[max_indx], thick)
                cv2.putText(imgcv, label, (left, top - 12),
                    0, 1e-3 * h, colors[max_indx],thick//3)
    else:
        detections = []
        scores = []

        global old_tracked_objects
        global tracked_objects
        old_tracked_objects = tracked_objects
        tracked_objects = {}

        start = current_milli_time()
        for b in boxes:
            boxResults = self.process_box(b, h, w, threshold)
            if boxResults is None:
                continue
            left, right, top, bot, label, max_indx, confidence = boxResults
            bbox = [left, top, right, bot]

            if label not in self.FLAGS.trackObj:
              continue
            if self.FLAGS.tracker == "deep_sort":
                detections.append(np.array([left,top,right-left,bot-top]).astype(np.float64))
                scores.append(confidence)
            elif self.FLAGS.tracker == "sort":
                detections.append(np.array([left, top, right, bot, confidence, label], dtype=object))
        end = current_milli_time()

        time_elapsed = (end - start) / 1000
        #TODO: remove this
        #print("running self.process_box on all detections took: {}".format(time_elapsed))
        if len(detections) < 3  and self.FLAGS.BK_MOG:
            detections = detections + extract_boxes(self,mask)

        detections = np.array(detections)

        if detections.shape[0] == 0 :
            return imgcv, None
        if self.FLAGS.tracker == "deep_sort" and tracker != None:
            scores = np.array(scores)
            features = encoder(imgcv, detections.copy())
            detections = [
                        Detection(bbox, score, feature) for bbox,score, feature in
                        zip(detections,scores, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = prep.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            tracker.predict()
            tracker.update(detections)
            trackers = tracker.tracks
        elif self.FLAGS.tracker == "sort" and tracker != None:
            start = current_milli_time()
            trackers = tracker.update(detections)
            end = current_milli_time()

            time_elapsed = (end - start) / 1000
            #TODO: remove this
            #print("sort_tracker.update(detections) took: {}".format(time_elapsed))

        if tracker != None:
            start = current_milli_time()
            for track in trackers:
                label = ""
                bbox = []
                if self.FLAGS.tracker == "deep_sort":
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    id_num = str(track.track_id)
                elif self.FLAGS.tracker == "sort":
                    bbox = [int(track[0]),int(track[1]),int(track[2]),int(track[3])]
                    id_num = str(track[-1])
                    label = track[5]

                if self.FLAGS.csv:
                    csv.writerow([frame_id,id_num,int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])])
                    csv_file.flush()

                object_key = label + " " + id_num
                tracked_objects[object_key] = {
                    "bbox": {
                        "topleft": {
                            "x": bbox[0], 
                            "y": bbox[1]
                        }, 
                        "bottomright": {
                            "x": bbox[2],
                            "y": bbox[3]
                        }
                    }
                }

                if object_key in old_tracked_objects and "person_name" in old_tracked_objects[object_key]:
                    tracked_objects[object_key]["person_name"] = old_tracked_objects[object_key]["person_name"]
                    label = tracked_objects[object_key]["person_name"]
                else:
                    if label == "person":
                        person_name_label = self.get_label_for_person(faces, labels, bbox)
                        if person_name_label is not None:
                            tracked_objects[object_key]["person_name"] = person_name_label
                            label = person_name_label

                if (self.FLAGS.display or self.FLAGS.saveVideo):
                    cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                    (255,255,255), thick//3)
                    cv2.putText(imgcv, label, (int(bbox[0]), int(bbox[1]) - 12),0, 1e-3 * h, (255,255,255),thick//6)
            end = current_milli_time()

            time_elapsed = (end - start) / 1000
            #TODO: remove this
            #print("drawing all tracked objects took: {}".format(time_elapsed))

            start = current_milli_time()
            speech_actions = object_detection_speech(self.FLAGS.speech, tracked_objects, old_tracked_objects, h, w)
            end = current_milli_time()

            time_elapsed = (end - start) / 1000
            #TODO: remove this
            #print("object detection and speech took: " + str(time_elapsed))

            if len(speech_actions) > 0:
                speech_actions = {
                    "isStart": False,
                    "isEnd": False,
                    "timestamp": int(round(timer())),
                    "video_id": video_id,
                    "actions": speech_actions
                }

            if self.FLAGS.upload and len(speech_actions) > 0:
                socketio_json = {
                    "isStart": False,
                    "isEnd": False,
                    "timestamp": int(round(timer())),
                    "video_id": video_id,
                    "actions": speech_actions
                }
                with SocketIO("http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com", 80, LoggingNamespace) as socketIO:
                    socketIO.emit("video_data_point", socketio_json)

    #Sidewalk Detection
    if self.FLAGS.sidewalk_detection:
        _, command = run_sidewalk_detection(im)
        #print("navigation command: " + command)

    return imgcv, speech_actions
