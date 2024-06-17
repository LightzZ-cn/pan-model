import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import os.path as osp
import os
import torch

defalut_lrms_path = ''
defalut_ms_path = ''
defalut_pan_path = ''
class PanMsDataset(data.Dataset):
    def __init__(self, lrms_path=defalut_lrms_path, pan_path=defalut_pan_path, ms_path=defalut_ms_path):
        super(PanMsDataset,self).__init__()
        self.ms_path = ms_path
        self.pan_path = pan_path
        self.lrms_path = lrms_path

        self.lrms = []
        self.ms = []
        self.pan = []
        
        for ms_name in os.listdir(ms_path):
           ms = Image.open(osp.join(self.ms_path,ms_name))
           ms = np.array(ms)
           ms = torch.tensor(ms).permute(2,1,0)
           self.ms.append(ms)

        for lr_ms_name in os.listdir(lrms_path):
            lr_ms = Image.open(osp.join(self.lrms_path,lr_ms_name))
            lr_ms = lr_ms.resize((lr_ms.width*2, lr_ms.height*2),resample=Image.BICUBIC)
            lr_ms = np.array(lr_ms)
            lr_ms = torch.tensor(lr_ms).permute(2,1,0)
            self.lrms.append(lr_ms)
        
        for pan_name in os.listdir(pan_path):
            pan = Image.open(osp.join(self.pan_path,pan_name))
            pan = np.array(pan)
            pan = torch.tensor(pan).permute(2,1,0)
            self.pan.append(pan)



    def __len__(self):

        return (len(os.listdir(self.ms_path)))
    
    def __getitem__(self, idx):

        p = self.pan[idx].float().cuda()
        lrms = self.lrms[idx].float().cuda()
        ms = self.ms[idx].float().cuda()

        return p , lrms , ms


        
