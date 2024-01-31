import cv2, torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

class Building_Frames():
    def __init__(self, path_video):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                                                  weights='DEFAULT')
        self.model = self.model.eval().to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.path_video = path_video
        self.cos_sim_lst = []
        self.wgh_sim_lst = []
        self.FI = cv2.FONT_ITALIC
        print(f"\n device: {self.device}\n")
        self.coord_metrics_lst = [(900, 900), (480, 1440), (320, 820, 1360), 
                    (150, 580, 1020, 1440), 
                    (20, 384, 776, 1150, 1540),
                    (20, 384, 776, 1150, 1540, 1540),
                    (20, 384, 776, 1150, 1540, 1540),
                    (20, 384, 776, 1150, 1540, 1540),
                    (20, 384, 776, 1150, 1540, 1540)]

    def Data_Video(self):
        cap = cv2.VideoCapture(self.path_video)
        frameRate = cap.get(5) 
        in_frame = []
        while(cap.isOpened()):
            frameId = cap.get(1) 
            ret, frame = cap.read()
            if (ret != True):
                break
            else:
                in_frame.append(frame)
        
        # Characteristics of video frames.  
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fn = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        print(f" file: {self.path_video}")
        print(f" width: {w}")
        print(f" height: {h}")
        print(f" fps: {fps}")
        print(f" frame count: {fn}")
        return w, h, fps, fn, in_frame

    def Data_Values(self):
        keypoints = ['nose', 'left_eye', 'right_eye',
                        'left_ear', 'right_ear', 'left_shoulder',
                        'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip',
                        'right_hip',  'left_knee',   'right_knee',
                        'left_ankle', 'right_ankle']

        def get_limbs_from_keypoints(keypoints):
            limbs = [
            [keypoints.index("right_eye"), keypoints.index("nose")],
            [keypoints.index("right_eye"), keypoints.index("right_ear")],
            [keypoints.index("left_eye"), keypoints.index("nose")],
            [keypoints.index("left_eye"), keypoints.index("left_ear")],
            [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
            [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
            [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
            [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
            [keypoints.index("right_hip"), keypoints.index("right_knee")],
            [keypoints.index("right_knee"), keypoints.index("right_ankle")],
            [keypoints.index("left_hip"), keypoints.index("left_knee")],
            [keypoints.index("left_knee"), keypoints.index("left_ankle")],
            [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
            [keypoints.index("right_hip"), keypoints.index("left_hip")],
            [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
            [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
            ]
            return limbs

        self.limbs = get_limbs_from_keypoints(keypoints)
        self.color_lst = [(255, 255, 255), (178, 242, 149), (94, 249, 197),
        (12, 193, 231), (106, 252, 191), (224, 210, 119),
        (252, 183, 99), (190, 236, 142), (150, 252, 167),
        (72, 240, 208)]
        self.dance_lst = ['MODEL', 'DANC-'] 

        return keypoints, self.limbs, self.color_lst, self.dance_lst

    # Recording text in a video frame.   
    def img_text(self, f_img, f_txt, coord, f_face=cv2.FONT_HERSHEY_SIMPLEX, 
                 f_scale=0.9, f_color=(0,0,255), th=2):
        cv2.putText(img = f_img, text = f_txt, org = coord,
            fontFace = f_face, fontScale = f_scale,  
            color = f_color, thickness=th)

    # Converting a tensor matrix into an array.  
    def conv(self, array, th):
        mask_obj = array['scores'] > th
        for i in ['boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores']:
            array[i] = array[i][mask_obj]
            if str(type(array[i])) != "<class 'numpy.ndarray'>":
                array[i] = array[i].cpu().numpy()

        return array

    # Preparing a video file for processing.  
    def correction(self, in_lst, th, flop=0):
        tmp_frame, tmp_skelet = [], []
        for nf, img in enumerate(in_lst):
            # Transformation - obtaining an image tensor.  
            img_tensor = self.transform(img).to(self.device)
            # Running the image through the model.  
            with torch.no_grad():
               img_out =self.model([img_tensor])[0]

            # Threshold filtering and conversion to np.array.  
            img_out = self.conv(img_out, th)

            # Removing negative tensor values.  
            mask_neg = img_out['keypoints_scores']
            img_out['keypoints_scores'][mask_neg < 0] = 0

            # Deleting network labels. 
            img_out['keypoints'] = img_out['keypoints'][:,:,:2]

            # The number of dancers.  
            if flop == 0:
                self.num_dancers = len(img_out['keypoints'])
                print(f"The number of dancers: {self.num_dancers}")
                print("\nPlease wait!\n")
                flop = 1
            
            # Skipping a frame if the number of dancers changes.   
            if self.num_dancers != len(img_out['keypoints']):
                continue

            # Adding the necessary keys and values.   
            img_out['names'] = np.array(
                [i for i in range(img_out['keypoints'].shape[0])]).astype(str)
            img_out['coord'] = np.array(
                [i for i in range(img_out['keypoints'].shape[0])]).astype(tuple)
            img_out['color'] = np.array(
                [255 for i in range(img_out['keypoints'].shape[0])]).astype(tuple)
            img_out['all_keys'] = img_out['keypoints'].copy()
            img_out['all_scores'] = img_out['keypoints_scores'].copy()

            for n, person_id in enumerate(
                        img_out['keypoints'][:,:,0].mean(axis=1).argsort()):

                # Eliminating the "jumping" of objects in the model.    
                kp = img_out['keypoints'][person_id]
                ks = img_out['keypoints_scores'][person_id]
                img_out['all_keys'][n] = kp
                img_out['all_scores'][n] = ks

                # The names of the dancers.  
                name = 'MODEL' if person_id==0 else 'DANC-'+str(person_id)
                img_out['names'][person_id] = name

                # The coordinates of the names.   
                keypoints = img_out['keypoints'][person_id, ...]
                keys = tuple(map(int, keypoints[0, :2]))
                img_out['coord'][n] = (keys[0]-50, keys[1]-50)
                if self.num_dancers > 5:
                    img_out['color'][person_id] = self.color_lst[2]
                else:
                    img_out['color'][person_id] = self.color_lst[person_id]

            img_out['keypoints'] = img_out['all_keys']
            img_out['keypoints_scores'] = img_out['all_scores']
            del img_out['all_keys']
            del img_out['all_scores']

            skeletal, cos_sim_lst, wgh_sim_lst = self.draw_skeleton_per_person(
                img, 
                img_out['keypoints'],
                img_out["keypoints_scores"], 
                img_out["scores"],
                img_out['names'],
                img_out['boxes'],
                img_out['coord'],
                img_out['color']) 

            tmp_skelet.append(skeletal)

        return tmp_skelet, cos_sim_lst, wgh_sim_lst, self.num_dancers

    def affine_transform(self, ref_keys, tst_keys, ref_confs, tst_confs):
        ref_keys = ref_keys
        tst_keys = tst_keys

        # pad and unpad to add or remove 1 at the end of the matrix.  
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]

        # Expanding arrays by adding 1.  
        X = pad(tst_keys)
        Y = pad(ref_keys)
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

        # Converting too small values to "0".   
        A[np.abs(A) < 1e-10] = 0

        # Now that we have found the extended matrix A,
        # we can transform the input set of key points.  
        transform = lambda x: unpad(np.dot(pad(x), A))
        keypoints_transformed = transform(tst_keys)
        return keypoints_transformed

    # Centering function with the conversion of an array into a vector.    
    def center(self, Xar, F=True):
        newX = Xar - np.mean(Xar, axis = 0)
        newX = newX.flatten() if F else newX
        return newX

    # The cosine similarity calculation function.    
    def cosine_similarity(self, pose1, pose2):
        pose1, pose2 = [self.center(i) for i in [pose1, pose2]]
        return np.dot(pose1, pose2.T) / \
        (np.linalg.norm(pose1)*np.linalg.norm(pose2))


    # Weighted cosine similarity calculation function.  
    def weighted_distance(self, pose1, pose2, confs):
        # Centering.  
        pose1, pose2 = [self.center(i, False) for i in [pose1, pose2]]
        # Normalization.  
        pose1, pose2 = [i/np.linalg.norm(i) for i in [pose1, pose2]]
        # Summation of weighted distances between keypoints.  
        sum_v = 0
        for k in range(len(pose1)):
            sum_v += (confs[k] 
            * np.linalg.norm(pose1[k]-pose2[k]))
        return sum_v / confs.sum()

    def draw_skeleton_per_person(self, img, all_keypoints, 
                all_scores, confs, 
                txt, box, coord, color, 
                keypoint_threshold=2):    

        img_copy = img.copy()
        tmp_lst = []
        tmp_wgh_lst = []
        for n_id, person_id in enumerate(
                all_keypoints[:,:,0].mean(axis=1).argsort()):
            keypoints_transformed = self.affine_transform(
                all_keypoints[0], all_keypoints[person_id], 
                confs[0], confs[person_id])
            cos_sim = round(self.cosine_similarity(
                keypoints_transformed, all_keypoints[0])*100, 4)
            W = np.sqrt(all_scores[0] * all_scores[person_id])
            wgh_sim = round((1-self.weighted_distance(
                keypoints_transformed, all_keypoints[0], W))*100, 4)
            tmp_lst.append(cos_sim)
            tmp_wgh_lst.append(wgh_sim)
            if len(confs) == n_id+1:
                self.cos_sim_lst.append(tmp_lst)
                self.wgh_sim_lst.append(tmp_wgh_lst)

            # Position X and the output of the dancer's name.   
            if self.num_dancers < 6:
                x_pos = self.coord_metrics_lst[len(confs)-1][person_id]
                self.img_text(img_copy, txt[person_id], coord[person_id], 
                              self.FI, 1.15, color[person_id])
                txt_cos = "Cosin. sim %: "+str(cos_sim)
                txt_wgh = "Weigh. sim %: "+str(wgh_sim)
            else:
                txt_cos, txt_wgh = '', ''
                x_pos = 0

            # Shifts along the axes to get a shadow.    
            shf_x = 3
            shf_y = -2

            # The text of cosine similarity values.    
            self.img_text(img_copy, txt_cos, (x_pos, 1035))
            self.img_text(img_copy, txt_cos, (x_pos-shf_x, 1035+shf_y), 
                          f_color=(51,254,254))

            # The text of the weighted cosine similarity values.    
            self.img_text(img_copy, txt_wgh, (x_pos, 1070))       
            self.img_text(img_copy, txt_wgh, (x_pos-shf_x, 1070+shf_y), 
                          f_color=(51,254,254))

            keypoints = all_keypoints[person_id, ...]
            scores = all_scores[person_id, ...]

            # The construction of points and lines on the figure of the dancer.    
            for kp in range(len(scores)):
                if scores[kp] > keypoint_threshold:
                    keypoint = tuple(map(int, keypoints[kp, :2]))
                    cv2.circle(img_copy, keypoint, 5, color[person_id], -1)
            
            for limb_id in range(len(self.limbs)):
                limb_loc1 = tuple(map(int, keypoints[self.limbs[limb_id][0], :2]))
                limb_loc2 = tuple(map(int, keypoints[self.limbs[limb_id][1], :2]))
                limb_score = min(all_scores[person_id, 
                             self.limbs[limb_id][0]], all_scores[person_id, 
                             self.limbs[limb_id][1]])
                if limb_score> keypoint_threshold:
                    cv2.line(img_copy, limb_loc1, limb_loc2, 
                             color[person_id], thickness=2)
        return img_copy, self.cos_sim_lst, self.wgh_sim_lst