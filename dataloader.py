import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T


class DataLoader(data.Dataset):
    def __init__(self, video_folder, img_size, frame_len, video_names, au_folder=None):

        self.dir = video_folder
        self.AUs_root = au_folder if au_folder is not None else '/data01/behavior_group/d21_qiaoyy/DOLOs/AU/4f/'
        self.frame_len = frame_len
        self.Max_len = 35
        self.video_names = video_names

        self.tgt_au = [14, 11, 3, 7, 6, 15]

        self.transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for video_name in video_names:
            video_path = os.path.join(self.dir, video_name)
            if os.path.exists(video_path):
                img_names = [i for i in os.listdir(video_path) if i.endswith('.jpg')]
                if len(img_names) < 60:
                    print(f"[Warning] {video_name} has only {len(img_names)} frames")

    def __getitem__(self, index):
        video_name = self.video_names[index]

        if "lie" in video_name or "deception" in video_name:
            label = 1
        elif "truth" in video_name or "true" in video_name:
            label = 0
        else:
            raise ValueError(f"Cannot parse label from video name: {video_name}")

        AUs_path = os.path.join(self.AUs_root, video_name + '.npy')
        raw_AUs = np.load(AUs_path, allow_pickle=True)

        AUs = []
        for n in self.tgt_au:
            au_events = raw_AUs[n]
            if au_events.size == 0 or au_events.shape[0] == 0:
                AUs.append(np.zeros((self.Max_len, 4), dtype=np.float32))
            else:
                au_events = np.asarray(au_events, dtype=np.float32)
                if au_events.shape[0] < self.Max_len:
                    pad = np.zeros((self.Max_len - au_events.shape[0], 4), dtype=np.float32)
                    au_events = np.vstack([au_events, pad])
                elif au_events.shape[0] > self.Max_len:
                    au_events = au_events[:self.Max_len]
                AUs.append(au_events.copy())

        # (n_parts=6, Max_len=35, D=4)
        AUs = np.stack(AUs, axis=0).astype(np.float32)

        video_path = os.path.join(self.dir, video_name)
        img_names = sorted([i for i in os.listdir(video_path) if i.endswith('.jpg')])
        vlen = len(img_names)

        target_indices = np.linspace(0, vlen - 1, num=self.frame_len)
        target_indices = np.around(target_indices).astype(int)

        img_samples = []
        for idx in target_indices:
            img = Image.open(os.path.join(video_path, img_names[idx])).convert('RGB')
            img_samples.append(self.transforms(img))

        img_samples = torch.stack(img_samples, dim=0).permute(1, 0, 2, 3).contiguous()

        return img_samples, AUs, label, video_name

    def __len__(self):
        return len(self.video_names)