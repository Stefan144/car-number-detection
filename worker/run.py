import cv2

cv2.setNumThreads(0)

import argparse
import logging
from logging.handlers import RotatingFileHandler

from worker.state import State
from worker.video_reader import VideoReader
from worker.ocr_stream import OcrStream
from worker.visualize_stream import VisualizeStream


def setup_logging(path, level='INFO'):
    handlers = [logging.StreamHandler()]
    file_path = path
    if file_path:
        file_handler = RotatingFileHandler(filename=file_path,
                                           maxBytes=10 * 10 * 1024 * 1024,
                                           backupCount=5)
        handlers.append(file_handler)
    logging.basicConfig(
        format='[{asctime}][{levelname}] - {name}: {message}',
        style='{',
        level=logging.getLevelName(level),
        handlers=handlers,
    )


class CNDProject:
    def __init__(self, name, video_path, save_path, model_path, car_number,
                 fps=30, frame_size=(1600, 800), coord=(500, 500)):
        self.model_path = model_path
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = State()
        self.video_reader = VideoReader("VideoReader", video_path)
        self.ocr_stream = OcrStream("OcrStream", self.state, self.video_reader,
                                    self.model_path)

        self.visualize_stream = VisualizeStream("VisualizeStream", self.video_reader,
                                                self.state, save_path, car_number, fps,
                                                frame_size, coord)
        self.logger.info("Start Project")

    def start(self):
        self.logger.info("Start project act start")
        try:
            self.video_reader.start()
            self.ocr_stream.start()
            self.visualize_stream.start()
            self.state.exit_event.wait()
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stop Project")

        self.video_reader.stop()
        self.ocr_stream.stop()
        self.visualize_stream.stop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", help="path to log", required=True)
    parser.add_argument("--level", help="logging level", default="INFO")
    parser.add_argument("--video_path", help="path to video", required=True)
    parser.add_argument("--save_path", help="path to save the new video",
                        required=True)
    # 'model-163-0.351443.pth' or  'model-250-0.020266.pth'
    parser.add_argument("--model_path", help="path to the model", required=True)
    parser.add_argument("--car_number", help="target value", required=True)
    args = parser.parse_args()

    setup_logging(args.log_path, args.level)
    logger = logging.getLogger(__name__)
    project = None
    try:
        project = CNDProject("CNDProject", args.video_path, args.save_path,
                             args.model_path, args.car_number)
        project.start()
    except Exception as e:
        logger.exception(e)
    finally:
        if project is not None:
            project.stop()
