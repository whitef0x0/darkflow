"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
from time import time as timer
import tensorflow as tf
import numpy as np
import sys
import cv2
import os
import csv
import uuid
import requests
import time
from socketIO_client import SocketIO, LoggingNamespace
current_milli_time = lambda: int(round(timer() * 1000))

old_graph_msg = 'Resolving old graph def {} (no guarantee)'

def build_train_op(self):
    self.framework.loss(self.out)
    self.say('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    gradients = optimizer.compute_gradients(self.framework.loss)
    self.train_op = optimizer.apply_gradients(gradients)

def load_from_ckpt(self):
    if self.FLAGS.load < 0: # load lastest ckpt
        with open(self.FLAGS.backup + 'checkpoint', 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.FLAGS.load = int(load_point)

    load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
    load_point = '{}-{}'.format(load_point, self.FLAGS.load)
    self.say('Loading from {}'.format(load_point))
    try: self.saver.restore(self.sess, load_point)
    except: load_old_graph(self, load_point)

def say(self, *msgs):
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        print(msg)

def load_old_graph(self, ckpt):
    ckpt_loader = create_loader(ckpt)
    self.say(old_graph_msg.format(ckpt))

    for var in tf.global_variables():
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        assert val is not None, \
        'Cannot find and load {}'.format(var.name)
        shp = val.shape
        plh = tf.placeholder(tf.float32, shp)
        op = tf.assign(var, plh)
        self.sess.run(op, {plh: val})

def _get_fps(self, frame):
    elapsed = int()
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame)
    return timer() - start

def setup_camera(self):
    self.video_id = str(uuid.uuid4())

    if self.FLAGS.track :
        if self.FLAGS.tracker == "deep_sort":
            from deep_sort import generate_detections
            from deep_sort.deep_sort import nn_matching
            from deep_sort.deep_sort.tracker import Tracker
            metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", 0.2, 100)
            self.tracker = Tracker(metric)
            self.encoder = generate_detections.create_box_encoder(
                os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
        elif self.FLAGS.tracker == "sort":
            from sort import sort
            self.encoder = None
            self.tracker = sort.Sort()
    if self.FLAGS.BK_MOG and self.FLAGS.track :
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    self.camera = cv2.VideoCapture(0)
    cam_h2w = 720/1280
    expected_width = 640
    expected_height = expected_width * cam_h2w
    self.camera.set(3,expected_height)
    self.camera.set(4,expected_width)

    assert self.camera.isOpened(), 'Cannot capture source'

    f = None
    writer = None

    cv2.startWindowThread()
    cv2.namedWindow('LiveFeed', 0)
    _, frame = self.camera.read()
    self.frame_height, self.frame_width, _ = frame.shape
    cv2.resizeWindow('LiveFeed', self.frame_width, self.frame_height)

    if self.FLAGS.saveVideo:
        self.videoWriter = cv2.VideoWriter('output_video.mov', -1, 2, (self.frame_width, self.frame_height))

    # buffers for demo in batch
    self.buffer_inp = list()
    self.buffer_pre = list()

    self.elapsed = 0
    self.fps_start = timer()

    if self.FLAGS.upload:
        socketio_json = {
            "isStart": True,
            "isEnd": False,
            "timestamp": int(round(time.time())),
            "video_id": self.video_id,
            "actions": []
        }
        with SocketIO('http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com', 80, LoggingNamespace) as socketIO:
            socketIO.emit('video_data_point', socketio_json)

    self.frame_grayscale = None
    self.n = 0

    return self.camera

def process_frame(self):
    self.elapsed += 1
    _, frame = self.camera.read()

    if frame is None:
        return True
    if self.FLAGS.skip != self.n :
        self.n+=1
        return False

    previous_frame = self.frame_grayscale
    self.frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    self.n = 0
    if self.FLAGS.BK_MOG and self.FLAGS.track :
        fgmask = fgbg.apply(frame)
    else:
        fgmask = None

    start = current_milli_time()

    preprocessed = self.framework.preprocess(frame)

    end = current_milli_time()
    time_elapsed = (end - start) / 1000
    #TODO: remove this
    #print("self.framework.preprocess(...) took: {}".format(time_elapsed))

    self.buffer_inp.append(frame)
    self.buffer_pre.append(preprocessed)
    # Only process and imshow when queue is full
    if self.elapsed % self.FLAGS.queue == 0:
        feed_dict = {self.inp: self.buffer_pre}
        net_out = self.sess.run(self.out, feed_dict)
        for img, single_out in zip(self.buffer_inp, net_out):
            postprocessed = None
            if not self.FLAGS.track:
                postprocessed = self.framework.postprocess(
                    single_out, img)
            else:
                start = current_milli_time()

                postprocessed = self.framework.postprocess(
                    single_out, img, self.video_id, frame_id=self.elapsed,
                    csv_file=None, csv=None, mask=None,
                    encoder=self.encoder, tracker=self.tracker, previous_frame=previous_frame)
            
                end = current_milli_time()
                time_elapsed = (end - start) / 1000
                #TODO: remove this
                #print("self.framework.postprocess(...) took: {}".format(time_elapsed))

            if self.FLAGS.saveVideo:
                start = current_milli_time()

                self.videoWriter.write(postprocessed)

                end = current_milli_time()
                time_elapsed = (end - start) / 1000
                #TODO: remove this
                #print("videoWriter.write(postprocessed) took: {}".format(time_elapsed))

            if self.FLAGS.display:
                cv2.imshow('LiveFeed', postprocessed)

        # Clear Buffers
        self.buffer_inp = list()
        self.buffer_pre = list()

    if self.elapsed % 5 == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{0:3.3f} FPS'.format(
            self.elapsed / (timer() - self.fps_start)))
        sys.stdout.flush()

    if self.FLAGS.saveVideo:
        start = current_milli_time()

        self.videoWriter.write(postprocessed)

        end = current_milli_time()
        time_elapsed = (end - start) / 1000
        #TODO: remove this
        #print("videoWriter.write(postprocessed) took: {}".format(time_elapsed))

    #if self.FLAGS.display:
    choice = cv2.waitKey(1)

    return False

def teardown_camera(self):
    if self.FLAGS.upload:
        socketio_json = {
            "isStart": False,
            "isEnd": True,
            "timestamp": int(round(time.time())),
            "video_id": self.video_id,
            "actions": []
        }
        with SocketIO('http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com', 80, LoggingNamespace) as socketIO:
            socketIO.emit('video_data_point', socketio_json)

    sys.stdout.write('\n')

    if self.FLAGS.saveVideo:
        self.videoWriter.release()

    if self.FLAGS.upload:
        url = 'http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com/video_stream/' + self.video_id
        files = {'file': open('output_video.mov', 'rb')}
        r = requests.post(url, files=files)
        os.remove('output_video.mov')

    self.camera.release()
    if self.FLAGS.display :
        cv2.destroyAllWindows()

def camera(self):
    if self.FLAGS.track :
        if self.FLAGS.tracker == "deep_sort":
            from deep_sort import generate_detections
            from deep_sort.deep_sort import nn_matching
            from deep_sort.deep_sort.tracker import Tracker
            metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", 0.2, 100)
            tracker = Tracker(metric)
            encoder = generate_detections.create_box_encoder(
                os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
        elif self.FLAGS.tracker == "sort":
            from sort import sort
            encoder = None
            tracker = sort.Sort()
    if self.FLAGS.BK_MOG and self.FLAGS.track :
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    camera = cv2.VideoCapture(0)
    cam_h2w = 720/1280
    expected_width = 640
    expected_height = expected_width * cam_h2w
    camera.set(3,expected_height)
    camera.set(4,expected_width)

    self.say('Press [ESC] to quit video')

    assert camera.isOpened(), 'Cannot capture source'

    f = None
    writer = None

    cv2.startWindowThread()
    cv2.namedWindow('LiveFeed', 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('LiveFeed', width, height)

    if self.FLAGS.saveVideo:
        videoWriter = cv2.VideoWriter('output_video.mov', -1, 2, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()

    elapsed = 0
    fps_start = timer()
    self.say('Press [ESC] to quit demo')

    # Loop through frames
    n = 0
    video_id = str(uuid.uuid4())

    if self.FLAGS.upload:
        socketio_json = {
            "isStart": True,
            "isEnd": False,
            "timestamp": int(round(time.time())),
            "video_id": video_id,
            "actions": []
        }
        with SocketIO('http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com', 80, LoggingNamespace) as socketIO:
            socketIO.emit('video_data_point', socketio_json)

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()

        if frame is None:
            #print ('\nEnd of Video')
            break
        if self.FLAGS.skip != n :
            n+=1
            continue

        previous_frame = frame_grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        n = 0
        if self.FLAGS.BK_MOG and self.FLAGS.track :
            fgmask = fgbg.apply(frame)
        else :
            fgmask = None

        start = current_milli_time()

        preprocessed = self.framework.preprocess(frame)

        end = current_milli_time()
        time_elapsed = (end - start) / 1000
        #TODO: remove this
        #print("self.framework.preprocess(...) took: {}".format(time_elapsed))

        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                if not self.FLAGS.track:
                    postprocessed = self.framework.postprocess(
                        single_out, img)
                else:
                    start = current_milli_time()

                    postprocessed = self.framework.postprocess(
                        single_out, img, video_id, frame_id=elapsed,
                        csv_file=f, csv=writer, mask=fgmask,
                        encoder=encoder, tracker=tracker, previous_frame=previous_frame)
                
                    end = current_milli_time()
                    time_elapsed = (end - start) / 1000
                    #TODO: remove this
                    #print("self.framework.postprocess(...) took: {}".format(time_elapsed))
                if self.FLAGS.display:
                    cv2.imshow('LiveFeed', postprocessed)

                if self.FLAGS.saveVideo:
                    start = current_milli_time()

                    videoWriter.write(postprocessed)

                    end = current_milli_time()
                    time_elapsed = (end - start) / 1000
                    #TODO: remove this
                    #print("videoWriter.write(postprocessed) took: {}".format(time_elapsed))

            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - fps_start)))
            sys.stdout.flush()

        if self.FLAGS.display :
            choice = cv2.waitKey(1)

            #Check if we should quit
            if choice == 27: break


    if self.FLAGS.upload:
        socketio_json = {
            "isStart": False,
            "isEnd": True,
            "timestamp": int(round(time.time())),
            "video_id": video_id,
            "actions": []
        }
        with SocketIO('http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com', 80, LoggingNamespace) as socketIO:
            socketIO.emit('video_data_point', socketio_json)

    sys.stdout.write('\n')

    if self.FLAGS.saveVideo:
        videoWriter.release()

    if self.FLAGS.upload:
        url = 'http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com/video_stream/' + video_id
        files = {'file': open('output_video.mov', 'rb')}
        r = requests.post(url, files=files)
        #os.remove('output_video.mov')

    camera.release()
    if self.FLAGS.csv :
        f.close()
    if self.FLAGS.display :
        cv2.destroyAllWindows()

def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
        for var in tf.global_variables():
            name = var.name.split(':')[0]
            var_name = name.split('-')
            l_idx = int(var_name[0])
            w_sig = var_name[1].split('/')[-1]
            l = darknet_ckpt.layers[l_idx]
            l.w[w_sig] = var.eval(self.sess)

    for layer in darknet_ckpt.layers:
        for ph in layer.h:
            layer.h[ph] = None

    return darknet_ckpt
