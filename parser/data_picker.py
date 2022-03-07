import os
import json
import cv2
from PIL import Image

from tqdm.auto import tqdm

from ..param import MEMOR_ROOT, PARSED_DATA_ROOT


class MemorVideoPicker():
    def __init__(self, memor_root = MEMOR_ROOT, parsed_data_root = PARSED_DATA_ROOT) -> None:
        # load list of videos
        self.memor_root = memor_root
        self.parsed_data_root = parsed_data_root
        self.memor_video_folder = os.path.join(memor_root, "videos")
        # self.memor_videos = sorted(os.listdir(self.memor_video_folder))

        # load memor labeling data
        self.memor_data = json.load(open(os.path.join(self.memor_root, "data.json")))

        # video
        self.video_name = ""
        self.vidcap = None
        self.frame_rate = 24 

        # saving info
        # save image
        self.image_root = os.path.join(self.parsed_data_root, "memor_video_images")
        if not os.path.exists(self.image_root):
            os.mkdir(self.image_root)

    def load_video_clip(self, video_name:str):
        self.video_name = video_name
        # load video clup from video name
        video_path = os.path.join(self.memor_video_folder, video_name + ".mp4")
        self.vidcap = cv2.VideoCapture(video_path)

        # get video framerate
        self.frame_rate = self.vidcap.get(cv2.CAP_PROP_FPS)

    def get_and_save_video_image_at_time(self, frame_time:float, index:int):
        """
        Get one frame image and save
        :params:
            frame_time: time at video clip
            index: utterance index in the video
        """
        # to prevent overflow
        frames = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_at_key = self.frame_rate * frame_time
        frame_at_key = min(frame_at_key, frames - 1)

        # set video image at time
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_at_key)

        # load image at time
        success,image = self.vidcap.read()
        assert success

        # get rgb image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(image)

        # save image
        image_save_path = os.path.join(self.image_root, self.video_name, str(index) + ".jpg")
        pil_im.save(image_save_path)
        
    def parse_one_piece_of_data(self, video_name:str):
        video_info = self.memor_data[video_name]
        self.load_video_clip(video_name)
        
        save_path = os.path.join(self.image_root, self.video_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for idx in range(len(video_info["seg_ori_ind"])):
            # TODO: optimize video time, currently get middle
            video_at_time = (video_info["seg_start"][idx] + video_info["seg_end"][idx]) / 2.0 - \
                video_info["start"]

            self.get_and_save_video_image_at_time(video_at_time, idx)

    def parse_all_videos(self):
        all_video_names = sorted(self.memor_data)
        for video_name in tqdm(all_video_names):
            self.parse_one_piece_of_data(video_name)
        
if __name__ == "__main__":
    video_picker = MemorVideoPicker()
    video_picker.parse_all_videos()