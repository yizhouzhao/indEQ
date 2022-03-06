import os
import json
import cv2
from PIL import Image

# root of MEMOR data
MEMOR_ROOT = "/home/yizhou/Research/MEmor/MEmoR/data/"

# parsed video/image saving folder
PARSED_DATA_ROOT= "./parsed_data/"

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
        video_path = os.path.join(self.memor_video_folder, video_name)
        self.vidcap = cv2.VideoCapture(video_path)

        # get video framerate
        self.rame_rate = self.vidcap.get(cv2.CAP_PROP_FPS)

    def get_and_save_video_image_at_time(self, frame_time:float, index:int):
        """
        Get one frame image and save
        :params:
            frame_time: time at video clip
            index: utterance index in the video
        """
        # set video image at time
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_rate * frame_time)

        # load image at time
        success,image = self.vidcap.read()
        assert success

        # get rgb image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(image)

        # save image
        image_save_path = os.path.join(self.image_root, self.video_name + "_" + str(index) + ".jpg")
        pil_im.save(image_save_path)
        
    def parse_one_piece_of_data(self, video_name:str):
        video_info = self.memor_data[video_name]
        for idx in video_info["seg_ori_ind"]:
            video_at_time = 0

        
