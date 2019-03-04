import pyaudio
import wave
import struct
import numpy as np
from meross_iot.api import MerossHttpClient
import time
from gistfile1 import goertzel
from collections import deque
import multiprocessing
import regressor
import datetime
import secrets

client = MerossHttpClient(email=secrets.username, password=secrets.password)
devices = client.list_supported_devices()


state = [d.get_sys_data()['all']['control']['toggle']['onoff'] for d in devices]
state = max(state)


audio = pyaudio.PyAudio()
print(audio.get_device_info_by_index(0)['defaultSampleRate'])


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

#stream = audio.open(format=FORMAT, channels=CHANNELS,
#                rate=RATE, input=True,
#                frames_per_buffer=CHUNK)


frames = []
delay_active = False
delay_counter = 0
delay_amount = 10



class FeatureStream:

    def __init__(self, num_steps, ranges):
        self.num_steps = num_steps
        self.ranges = ranges
        #self.manager = multiprocessing.Manager()
        self.features = []
        self.callback_functions = []
        self.RATE = 44100
        self.CHUNK = 1024

        self.callback_functions = [] # should be of form (features --> ())

        #self.update_features_thread()
        #self.callback_functions = [] # should be of form (features --> ())

    def start(self):
        self.update_features_thread()




    def build_stream(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RECORD_SECONDS = 10
        WAVE_OUTPUT_FILENAME = "output.wav"

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
        return stream

    def grab_new_feature(self, stream):
        data = stream.read(self.CHUNK, exception_on_overflow=False)
        data = np.array(struct.unpack(f'{self.CHUNK}h', data))
        features = [x[2] for x in goertzel(data, self.RATE, *self.ranges)[1]]
        return features

    def make_feature_vector(self, features):
        vector = []
        for f in features:
            vector.extend(f)
        return vector

    def update_features_thread(self):
        stream = self.build_stream()
        while True:
            feature = self.grab_new_feature(stream)
            if len(self.features) >= self.num_steps:
                self.features = self.features[1:]
            self.features.append(feature)

            # run the callbacks
            vector = self.make_feature_vector(self.features)
            for c in self.callback_functions:
                c(vector)


    def register_callback(self, callback):
        self.callback_functions.append(callback)

def log_loud_noises(features):
    mag = np.mean(features)
    print(mag)
    CUTOFF = 1000000000000 / 1000
    if mag > CUTOFF:
        print('Got a loud noise!')
        with open('loud_noises.txt', 'a+') as f:
            f.write(','.join([str(x) for x in features])+'\n')

def log_claps(features):
    mag = np.mean(features)
    print(mag)
    CUTOFF = 1000000000000 / 1000
    if mag > CUTOFF:
        print('Got a loud noise!')
        with open('claps.txt', 'a+') as f:
            f.write(','.join([str(x) for x in features])+'\n')

clf = regressor.load_classifier()

def detect_clap(features):
    mag = np.mean(features)
    CUTOFF = 1000000000000 / 1000
    if mag < CUTOFF:
        return
    try:
        X = np.array([features])
        X /= 100000000000.
        is_clap = clf.predict(X)[0]
    except ValueError:
        return

    if is_clap:
        switch_light_state()


LAST_SWITCH = datetime.datetime.now()

def switch_light_state():
    global LAST_SWITCH
    state = [d.get_sys_data()['all']['control']['toggle']['onoff'] for d in devices]
    if datetime.datetime.now() - LAST_SWITCH < datetime.timedelta(seconds=2):
        print('Too Soon!')
        return
    if any(state):
        for d in devices:
            d.turn_off()
    else:
        for d in devices:
            d.turn_on()
    LAST_SWITCH = datetime.datetime.now()



fstream = FeatureStream(3, [(1600, 1800), (1800, 2200), (2200, 2400)])
fstream.register_callback(detect_clap)
fstream.start()