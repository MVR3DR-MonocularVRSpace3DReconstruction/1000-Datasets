# coding=utf-8
import os
import cv2
import glob
from pathlib import Path

data_dir = Path('inputs/classroom/')
depth_video = glob.glob(str(data_dir / "*depth.mp4"))
image_video = glob.glob(str(data_dir / "*image.mp4"))

def video2frames(input_dir, output_dir):
    os.system(f"rm -rf {str(data_dir / output_dir)}")
    os.system(f"mkdir {str(data_dir / output_dir)}")
    interval = 1 # 保存时的帧数间隔
    frame_count = 0 # 保存帧的索引
    frame_index = 0 # 原视频的帧索引，与 interval*frame_count = frame_index 
    cap = cv2.VideoCapture(input_dir)
    if cap.isOpened():
        success = True
    else:
        success = False
        print("Read failed!")

    while(success):
        success, frame = cap.read()
        if success is False:
            print("=> Frame-%d read failed // Read finish:" % frame_index)
            break
            
        print("=> Reading frame-%d:" % frame_index, success)
        
        if frame_index % interval == 0:
            cv2.imwrite(str(data_dir / output_dir / '{:0>5}.png'.format(frame_count)), frame)
            frame_count += 1
        frame_index += 1
        
video2frames(str(depth_video[0]), "depth")
video2frames(str(image_video[0]), "image")
