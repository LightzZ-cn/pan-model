


from submodel import Shared_Shallow_Encoder, Shared_Global_Encoder, MLIE, \
  FIIM, Shared_Decoder 

from dataSet import PanMsDataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia




os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# . Set the hyper-parameters for training
num_epochs = 3000 # total epoch

lr = 5e-4
weight_decay = 0
batch_size = 24
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
alpha = 0.9
beta = 0.1

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Shared_Shallow_Encode_i = nn.DataParallel(Shared_Shallow_Encoder()).to(device)
Shared_Global_Encoder_i = nn.DataParallel(Shared_Global_Encoder()).to(device)
Mlie_1 = nn.DataParallel(MLIE()).to(device)
Mlie_2 = nn.DataParallel(MLIE()).to(device)
Global_Information_Fusion_Layer_i = nn.DataParallel(Shared_Global_Encoder()).to(device)
Local_Information_Fusion_Layer_i = nn.DataParallel(MLIE()).to(device)
Global_FIIM = nn.DataParallel(FIIM()).to(device)
Local_FIIM = nn.DataParallel(FIIM()).to(device)
Shared_Decoder_i =nn.DataParallel(Shared_Decoder()).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(Shared_Shallow_Encode_i.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Shared_Global_Encoder_i.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(Mlie_1.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(Mlie_2.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(Global_Information_Fusion_Layer_i.parameters(), lr=lr, weight_decay=weight_decay)
optimizer6 = torch.optim.Adam(Local_Information_Fusion_Layer_i.parameters(), lr=lr, weight_decay=weight_decay)
optimizer7 = torch.optim.Adam(Global_FIIM.parameters(), lr=lr, weight_decay=weight_decay)
optimizer8 = torch.optim.Adam(Local_FIIM.parameters(), lr=lr, weight_decay=weight_decay)
optimizer9 = torch.optim.Adam(Shared_Decoder_i.parameters(), lr=lr, weight_decay=weight_decay)


MSELoss = nn.MSELoss()  
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


# data loader
trainloader = DataLoader(PanMsDataset(),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

Shared_Shallow_Encode_i.train()
Shared_Global_Encoder_i.train()
Mlie_1.train()
Mlie_2.train()
Global_Information_Fusion_Layer_i.train()
Local_Information_Fusion_Layer_i.train()
Global_FIIM.train()
Local_FIIM.train()
Shared_Decoder_i.train()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_PAN , data_LRMS , data_HRMS) in enumerate(loader['train']):
        data_VIS, data_IR, data_HRMS = data_VIS.cuda(), data_IR.cuda(), data_HRMS.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        optimizer7.zero_grad()
        optimizer8.zero_grad()
        optimizer9.zero_grad()

        
        f_1_lrms = Shared_Shallow_Encode_i(data_LRMS)
        f_1_pan = Shared_Shallow_Encode_i(data_PAN)
        
        f_2_lrms_l = Mlie_1(f_1_lrms)
        f_2_lrms_g = Shared_Global_Encoder_i(f_1_lrms)
        f_2_pan_l = Mlie_2(f_1_pan)
        f_2_pan_g = Shared_Global_Encoder_i(f_1_pan)

        f_3_g = Global_Information_Fusion_Layer_i(f_2_lrms_g,f_2_pan_g)
        f_3_g_c = Global_FIIM(f_2_lrms_g,f_2_pan_g)
        f_3_l= Local_Information_Fusion_Layer_i(f_2_lrms_l,f_2_pan_l)
        f_3_l_c = Local_FIIM(f_2_lrms_l,f_2_pan_l)

        f_4_g = f_3_g + f_3_g_c
        f_4_l = f_3_l + f_3_l_c

        result = Shared_Decoder_i(f_4_l, f_4_g)

        loss = alpha*MSELoss(result,data_HRMS)+beta*Loss_ssim(result,data_HRMS)

        loss.backward()
 
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()
        optimizer6.step()
        optimizer7.step()
        optimizer8.step()
        optimizer9.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    if optimizer1.param_groups[0]['lr'] <= 1e-6: optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6: optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6: optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6: optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6: optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6: optimizer6.param_groups[0]['lr'] = 1e-6
    if optimizer7.param_groups[0]['lr'] <= 1e-6: optimizer7.param_groups[0]['lr'] = 1e-6
    if optimizer8.param_groups[0]['lr'] <= 1e-6: optimizer8.param_groups[0]['lr'] = 1e-6
    if optimizer9.param_groups[0]['lr'] <= 1e-6: optimizer9.param_groups[0]['lr'] = 1e-6
    
if True:
    checkpoint = {
        'Shared_Shallow_Encode_i': Shared_Shallow_Encode_i.state_dict(),
        'Shared_Global_Encoder_i': Shared_Global_Encoder_i.state_dict(),
        'Mlie_2': Mlie_1.state_dict(),
        'Mlie_2': Mlie_2.state_dict(),
        'Global_Information_Fusion_Layer_i': Global_Information_Fusion_Layer_i.state_dict(),
        'Local_Information_Fusion_Layer_i.': Local_Information_Fusion_Layer_i.state_dict(),
        'Global_FIIM': Global_FIIM.state_dict(),
        'Local_FIIM': Local_FIIM.state_dict(),
        'Shared_Decoder_i':Shared_Decoder_i.state_dict()
    }
    torch.save(checkpoint, os.path.join("models/cp_"+timestamp+'.pth'))


