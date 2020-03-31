import logging
from threading import Thread
from cnd.ocr.transforms import get_transforms
import time

from cnd.ocr.predictor import Predictor
from worker.state import State
from worker.video_reader import VideoReader


class OcrStream:
    def __init__(self, name, state: State, video_reader: VideoReader, model_path):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.transform = get_transforms([32, 96])
        self.state = state
        self.video_reader = video_reader
        self.ocr_thread = None
        self.model_path = model_path
        self.predictor = Predictor(self.model_path)
        self.counter = 0
        self.stopped = False
        self.logger.info("Create OcrStream")

    def _ocr_loop(self):
        try:
            start_time = time.time()
            while True:
                if self.stopped:
                    return
                frame = self.video_reader.read()
                self.state.frame = frame
                frame = self.transform(frame)[None, :, :, :]
                pred = self.predictor.predict(frame)
                self.state.text = pred
                self.counter += 1
                if self.counter % 10 == 0:
                    self.logger.info('# frames processed: ' + str(self.counter))
                    self.logger.info('FPS rate: '
                                     + str(self.counter/(time.time() - start_time)))


        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def _start_ocr(self):
        self.ocr_thread = Thread(target=self._ocr_loop)
        self.ocr_thread.start()

    def start(self):
        self._start_ocr()
        self.logger.info("Start OcrStream")

    def stop(self):
        self.stopped = True
        if self.ocr_thread is not None:
            self.ocr_thread.join()
        self.logger.info("Stop OcrStream")
