import numpy as np
import torch, cv2, os
import matplotlib.pyplot as plt
from class_dance import Building_Frames

""" Set the required directory path and file name """

PATH_DIR = "video/"
NAME_FILE = "2_danc.mp4"

if os.path.isfile(PATH_DIR+NAME_FILE):
    print(f"\nProcessing the '{NAME_FILE}' file.")
else:
    print(f"\nFile '{NAME_FILE}' Not Found: No such file or directory\n")
    exit()

# The name of the output video file (created automatically).    
out_video_file = NAME_FILE.split('.')[0]+'_out.mp4'

TRESHOLD = 0.95
FHS = cv2.FONT_HERSHEY_SIMPLEX
FI = cv2.FONT_ITALIC

intro = Building_Frames(PATH_DIR+NAME_FILE)
model, device = intro.model, intro.device
v_width, v_height, fps, frame_count, in_frame_lst = intro.Data_Video()
keypoints_lst, limbs, color_lst, dance_lst = intro.Data_Values()
out, cos_sim_lst, wgh_sim_lst, num_obj = intro.correction( 
                                         in_frame_lst, TRESHOLD)
nar = np.array(cos_sim_lst)
nar_w = np.array(wgh_sim_lst)
cos_tot = []
# Calculation of the final results.    
# Choosing the best student.    
if num_obj > 1 and num_obj < 6:
    for i in range(num_obj):
        cos_tot.append(round(np.mean([nar[:,i], nar_w[:,i]]), 4))
        txt = f"MODEL : {cos_tot[i]}" if i==0 else f"DANC {i}: {cos_tot[i]}"
        y_pos_tot = 100 + (i*30)
        intro.img_text(out[-1], 'Total Scores %:', (1600, 40), FI, f_scale=1.2, th=4)
        intro.img_text(out[-1], txt, (1650, y_pos_tot), FHS, f_scale=0.7)
        intro.img_text(out[-1], txt, (1647, y_pos_tot-3), 
                       FHS, f_scale=0.7, f_color=(51,254,254))
    lst_ind = cos_tot.index(max(cos_tot[1:]))
    for i in range(0,6,3):
        color = (51,254,254) if i == 3 else (0,0,255)
        intro.img_text(out[-1], 'Best student: DANC '+str(lst_ind), 
                   (1600-i, y_pos_tot+(50-i)), FHS, f_color=color)

    # + 30 frames with results to the end of the frame list.    
    for i in range(30):
        out.append(out[-1])

# Recording the resulting video.    
video_out = cv2.VideoWriter(
    PATH_DIR+out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                 (v_width, v_height))
for frame in out:
    video_out.write(frame)
video_out.release()  
print(f"*** The video file '{out_video_file}' assembly is complete! ***\n")