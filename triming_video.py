# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:21:03 2022

@author: jonathan
"""

import os
#import cv2
import ffmpeg


entry_path = 'C:/Users/jonathan/Documents/Telecom 2A/IMA201/Projet/video/input2/fish.mp4'
output_path = 'fish_soup.mp4'
begin_trim = 0
ending_trim = 1

def trim(in_file, out_file, start, end):
    if os.path.exists(out_file):
        os.remove(out_file)
        
    #probe_result = ffmpeg.probe(in_file)
    #in_file_duration = probe_result.get("format", {}).get("duration",None)
    #print(in_file_duration)
    
    input_stream = ffmpeg.input(in_file)
    
    pts = "PTS-STARTPTS"
    video = input_stream.trim(start=start,end=end).setpts(pts)
    audio = (input_stream
             .filter_("atrim", start = start, end = end)
             .filter_("asetpts", pts))
    
    video_and_audio = ffmpeg.concat(video,audio,v=1,a=1)
    output = ffmpeg.output(video, out_file,format="mp4")
    output.run()

trim(entry_path,output_path,begin_trim,ending_trim)