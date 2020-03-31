import logging
import time
from threading import Thread
from Levenshtein import distance

import cv2
from worker.state import State
from worker.video_reader import VideoReader
from worker.video_writer import VideoWriter


class Visualizer:
    def __init__(self, state: State, coord, car_number, color=(255, 255, 255),
                 thick=5, font_scale=3, font=cv2.FONT_HERSHEY_SIMPLEX):
        self.state = state
        self.coord_x, self.coord_y = coord
        self.car_number = car_number
        self.color = color
        self.thickness = thick
        self.font_scale = font_scale
        self.font = font
        self.max_acc = 0
        self.dumb_metric = 0

    def _draw_ocr_text(self):
        frame = self.state.frame
        text = self.state.text

        cur_dumb_metric = len(set(text) & set(self.car_number)) # number of common chars
        if cur_dumb_metric > self.dumb_metric:
            self.dumb_metric = cur_dumb_metric

        min_len = min(len(self.car_number), len(text))
        n, match = 0, 0
        for i in range(min_len):
            if text[i] == self.car_number[i]:
                match += 1
            n += 1
        cur_acc = match/n if n != 0 else match
        if cur_acc > self.max_acc:
            self.max_acc = cur_acc

        if text:
            cv2.putText(frame, text,
                        (self.coord_x, self.coord_y),
                        self.font,
                        self.font_scale,
                        self.color,
                        self.thickness)
        return frame

    def __call__(self):
        frame = self._draw_ocr_text()
        return frame


class VisualizeStream:
    def __init__(self, name,
                 in_video: VideoReader,
                 state: State, video_path, car_number, fps, frame_size, coord):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.coord = coord
        self.fps = fps
        self.car_number = car_number
        self.frame_size = tuple(frame_size)

        self.out_video = VideoWriter("VideoWriter", video_path, self.fps,
                                     self.frame_size)
        self.sleep_time_vis = 0.05 / self.fps
        self.in_video = in_video
        self.stopped = True
        self.visualize_thread = None

        self.visualizer = Visualizer(self.state, self.coord, self.car_number)

        self.logger.info("Create VisualizeStream")

    def _visualize(self):
        try:
            while True:
                if self.stopped:
                    return
                frame = self.visualizer()
                if frame is None:
                    continue
                frame = cv2.resize(frame, self.frame_size)
                self.out_video.write(frame)

                time.sleep(self.sleep_time_vis)

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def start(self):
        self.logger.info("Start VisualizeStream")
        self.stopped = False
        self.visualize_thread = Thread(target=self._visualize, args=())
        self.visualize_thread.start()
        #self.in_video.start()

    def stop(self):
        self.logger.info("best str accuracy: " + str(self.visualizer.max_acc))
        self.logger.info("max number of common chars: " + str(self.visualizer.dumb_metric))
        self.logger.info("Stop VisualizeStream")
        self.stopped = True
        self.out_video.stop()
        if self.visualize_thread is not None:
            self.visualize_thread.join()
