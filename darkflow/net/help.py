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

def get_video_id(self):
    return self.video_id

def generate_video_id(self):
    self.video_id = str(uuid.uuid4())

def process_frame(self, frame, previous_frame, disable_facial=False):
    self.elapsed += 1

    if frame is None:
        return True
    if self.FLAGS.skip != self.n :
        self.n+=1
        return False, None

    if previous_frame is not None:
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

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

    postprocessed = None
    speech_actions = None

    self.buffer_inp.append(frame)
    self.buffer_pre.append(preprocessed)
    # Only process and imshow when queue is full
    if self.elapsed % self.FLAGS.queue == 0:
        feed_dict = {self.inp: self.buffer_pre}
        net_out = self.sess.run(self.out, feed_dict)
        for img, single_out in zip(self.buffer_inp, net_out):
            if not self.FLAGS.track:
                postprocessed = self.framework.postprocess(
                    single_out, img)
            else:
                start = current_milli_time()

                postprocessed, speech_actions = self.framework.postprocess(single_out, img, self.video_id, frame_id=self.elapsed, csv_file=None, csv=None, mask=None, encoder=self.encoder, tracker=self.tracker, previous_frame=previous_frame, disable_facial=disable_facial)
            
                end = current_milli_time()
                time_elapsed = (end - start) / 1000
                #TODO: remove this
                #print("self.framework.postprocess(...) took: {}".format(time_elapsed))

            if self.FLAGS.saveVideo and not disable_facial:
                start = current_milli_time()

                self.videoWriter.write(postprocessed)

                end = current_milli_time()
                time_elapsed = (end - start) / 1000
                #TODO: remove this
                #print("videoWriter.write(postprocessed) took: {}".format(time_elapsed))

            if self.FLAGS.display and disable_facial is not True:
                cv2.imshow('LiveFeed', postprocessed)

        # Clear Buffers
        self.buffer_inp = list()
        self.buffer_pre = list()

    if self.elapsed % 5 == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{0:3.3f} FPS'.format(
            self.elapsed / (timer() - self.fps_start)))
        sys.stdout.flush()

    if self.FLAGS.saveVideo and not disable_facial:
        start = current_milli_time()

        self.videoWriter.write(postprocessed)

        end = current_milli_time()
        time_elapsed = (end - start) / 1000
        #TODO: remove this
        #print("videoWriter.write(postprocessed) took: {}".format(time_elapsed))

    if self.FLAGS.display and not disable_facial:
        choice = cv2.waitKey(1)

    return False, speech_actions

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
