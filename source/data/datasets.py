import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from torchvision import transforms
import pandas as pd
import abc
from config import pose_dirs
# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


# bbox of hands



# use split rgb video for save time

# build base dataset
class Base_Dataset(Dataset.Dataset, abc.ABC):

    @abc.abstractmethod
    def __getitem__(self, index):
        pass
    
    @abc.abstractmethod
    def __len__(self):
        
        pass

    @abc.abstractmethod
    def load_pose(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
    
    def collate_fn(self, batch):
        tgt_batch,src_length_batch,name_batch,pose_tmp,gloss_batch = [],[],[],[],[]
        text_embed_list = []
        
        for item in batch:
            if len(item) == 6:
                name_sample, pose_sample, text, gloss, support_rgb_dict, text_embed = item
                text_embed_list.append(text_embed)
            else:
                name_sample, pose_sample, text, gloss, support_rgb_dict = item
                text_embed_list = None  # backward-compatible if text embeddings not used

            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch
                

        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch
        
        # ---- Text embedding padding + mask ----
        if text_embed_list is not None and len(text_embed_list) > 0:
            first_shape = text_embed_list[0].shape

            # Case 1: sequence embeddings [T, D]
            if len(first_shape) == 2:
                max_text_len = max(embed.shape[0] for embed in text_embed_list)
                dim = text_embed_list[0].shape[1]

                padded_text_embeds = []
                text_masks = []
                for emb in text_embed_list:
                    T = emb.shape[0]
                    pad_len = max_text_len - T
                    if pad_len > 0:
                        pad = emb[-1:].expand(pad_len, -1)  # repeat last token
                        padded = torch.cat([emb, pad], dim=0)
                    else:
                        padded = emb
                    padded_text_embeds.append(padded)
                    text_masks.append(torch.cat([torch.ones(T), torch.zeros(pad_len)]))

                tgt_input['text_embeddings'] = torch.stack(padded_text_embeds, 0)  # [B, T_t, D]
                tgt_input['text_mask'] = torch.stack(text_masks, 0).long()          # [B, T_t]

            # Case 2: single vector embeddings [D]
            elif len(first_shape) == 1:
                tgt_input['text_embeddings'] = torch.stack(text_embed_list, 0)      # [B, D]
                tgt_input['text_mask'] = torch.ones(len(text_embed_list), 1).long() # dummy mask [B, 1]
        
        return src_input, tgt_input


class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase
        self.text_embed_dir = './dataset/CSL_Daily/mt5_embeddings/'

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
            
        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)

        else:
            raise NotImplementedError

        self.list = list(self.raw_data.keys())

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.list)
        # return 10
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        # print(f"Processing sample: {name_sample}")
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
        # --- NEW: Load precomputed text embedding ---
        text_embed = None
        if self.text_embed_dir is not None:
            embed_path = os.path.join(self.text_embed_dir, self.phase, f"{name_sample}.pt")
            # print(f"Loading text embedding from: {embed_path}")
            if os.path.exists(embed_path):
                try:
                    data = torch.load(embed_path, map_location="cpu")
                    text_embed = data["embedding"].float()
                except Exception as e:
                    print(f"[Warning] Failed to load {embed_path}: {e}")

        if text_embed is not None:
            return name_sample, pose_sample, text, gloss, support_rgb_dict, text_embed
        else:
            return name_sample, pose_sample, text, gloss, support_rgb_dict

        # return name_sample,pose_sample,text, gloss, support_rgb_dict
    
    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
            
        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

# Large-scale pre-training corpora (single JSON, 99/1 train/holdout split).
LARGE_SCALE_DATASETS = {"CSL_News", "YT_ASL", "BOBSL"}


class S2T_Large_Scale_dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Large_Scale_dataset, self).__init__()
        self.args = args
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)

        if self.args.dataset in LARGE_SCALE_DATASETS:
            self.pose_dir = pose_dirs[args.dataset]
        else:
            raise NotImplementedError(self.args.dataset)
        sum_sample = len(self.annotation)
        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)
        
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, index):
        num_retries = 10  

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx:self.end_idx][index]

            text = sample['text']
            name_sample = sample['video']

            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])
    
            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break

        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return name_sample, pose_sample, text, _, support_rgb_dict
    
    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        
        duration = len(pose['scores'])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
            
        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        kps_with_scores = load_part_kp(skeletons, confs)
        
        support_rgb_dict = {}

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'
    
class S2T_Dataset_phoenix(Base_Dataset):
    def __init__(self, path, args, phase):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase
        self.text_embed_dir = './dataset/Phoenix/mt5_embeddings/'

        
        if "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
        else:
            self.pose_dir = pose_dirs[args.dataset]
            


        self.list = list(self.raw_data.keys())
        # print(f"Dataset {self.args.dataset} {phase} size: {len(self.list)}")

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.list)
        # return 50
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['name'])
        # print(sample)
        # print(text, name_sample)
        # print(pose_sample['body'].shape)
        # --- NEW: Load precomputed text embedding ---
        text_embed = None
        if self.text_embed_dir is not None:
            embed_path = os.path.join(self.text_embed_dir, f"{name_sample}.pt")
            # print(f"Loading text embedding from: {embed_path}")
            if os.path.exists(embed_path):
                try:
                    data = torch.load(embed_path, map_location="cpu")
                    text_embed = data["embedding"].float()
                except Exception as e:
                    print(f"[Warning] Failed to load {embed_path}: {e}")

        if text_embed is not None:
            return name_sample, pose_sample, text, gloss, support_rgb_dict, text_embed
        else:
            return name_sample, pose_sample, text, gloss, support_rgb_dict
    
    def load_pose(self, path):
        pose_path = os.path.join(self.pose_dir, path + '.h5')
        pose = utils.read_compressed_kp_with_index(pose_path)
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
            
        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_how2sign(Base_Dataset):
    def __init__(self, path, args, phase):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = pd.read_csv(path, sep="\t")
        self.phase = phase

        self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
        
        self.text_embed_dir = './dataset/How2Sign/mt5_embeddings/'
            


        # self.list = list(self.raw_data.keys())
        # print(f"Dataset {self.args.dataset} {phase} size: {len(self.list)}")

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.raw_data)
        # return 10
    
    def __getitem__(self, index):
        row = self.raw_data.iloc[index]
        sentence_name = row['SENTENCE_NAME']
        text = row['SENTENCE']
        gloss = ''

        pose_sample, support_rgb_dict = self.load_pose(sentence_name)
        text_embed = None
        if self.text_embed_dir is not None:
            embed_path = os.path.join(self.text_embed_dir, self.phase, f"{sentence_name}.pt")
            # print(f"Loading text embedding from: {embed_path}")
            if os.path.exists(embed_path):
                try:
                    data = torch.load(embed_path, map_location="cpu")
                    text_embed = data["embedding"].float()
                except Exception as e:
                    print(f"[Warning] Failed to load {embed_path}: {e}")

        if text_embed is not None:
            return sentence_name, pose_sample, text, gloss, support_rgb_dict, text_embed
        else:
            return sentence_name, pose_sample, text, gloss, support_rgb_dict

        # return sentence_name,pose_sample,text, gloss, support_rgb_dict
    
    def load_pose(self, path):
        pose_path = os.path.join(self.pose_dir, path + '.pkl')
        # pose = utils.read_compressed_kp_with_index(pose_path)
        with open(pose_path, 'rb') as f:
            pose = pickle.load(f)
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            # tmp = sorted(random.sample(range(duration), k=self.max_length))
            tmp = list(range(self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
            
        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_meindgs(Base_Dataset):
    def __init__(self, path, args, phase):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = pd.read_csv(path, sep="|")
        self.phase = phase
        self.text_embed_dir = './dataset/MeinDGS/mt5_embeddings/'

        
        if "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
        else:
            self.pose_dir = pose_dirs[args.dataset]
            


        # self.list = list(self.raw_data.keys())
        # print(f"Dataset {self.args.dataset} {phase} size: {len(self.list)}")

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.raw_data)
        # return 50
    
    def __getitem__(self, index):
        row = self.raw_data.iloc[index]
        video_index = row['index']
        video_name = row['video_name']
        video_name = video_name + f'_{video_index}'
        text = row['target']
        gloss = ''
        
        pose_sample, support_rgb_dict = self.load_pose(video_name)
        # print(sample)
        # print(text, name_sample)
        # print(pose_sample['body'].shape)
        # --- NEW: Load precomputed text embedding ---
        text_embed = None
        if self.text_embed_dir is not None:
            embed_path = os.path.join(self.text_embed_dir, self.phase, f"{video_name}.pt")
            # print(f"Loading text embedding from: {embed_path}")
            if os.path.exists(embed_path):
                try:
                    data = torch.load(embed_path, map_location="cpu")
                    text_embed = data["embedding"].float()
                except Exception as e:
                    print(f"[Warning] Failed to load {embed_path}: {e}")

        if text_embed is not None:
            return video_name, pose_sample, text, gloss, support_rgb_dict, text_embed
        else:
            return video_name, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self, path):
        pose_path = os.path.join(self.pose_dir, self.phase, path + '.h5')
        pose = utils.read_compressed_kp_with_index(pose_path)

        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0

        if duration > self.max_length:
            # tmp = sorted(random.sample(range(duration), k=self.max_length))
            tmp = list(range(self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
            
        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'
