from __future__ import print_function
import torch 
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from F1_UNET_V1_1 import UNetV1
from F5_JACCARD import Jaccard
#from F9_UNET_V2_4 import UNetV2
from F9_UNET_V2_3 import UNetV2
from F10_SEGNET_V1 import SegNet
from F11_SEGPLOT import segplot
from F12_DLINKNET_V3 import DinkNet101 
from F20_DILATEDUNET import CamDUNet
from F21_GENERAL_UNET import R2U_Net, AttU_Net, R2AttU_Net
from F22_NESTEDUNET import NestedUNet
from F23_DULANORM_UNET import DualNorm_Unet
from F24_INCEPTION_UNET import InceptionUNet
from F25_SCAG_UNET import AttU_Net_with_scAG
from F26_FSFNet import FSFNet
from F27_LMFFNet import LMFFNet
from LMFFNet_IREM import LMFFNet2
from F28_FASSDNet import FASSDNet
from F29_ENet import ENet
from F30_ELANet import ELANet
#from F14_DEEPLABV3PLUS_V1 import DeepLabv3_plus
#from F14_DEEPLABV3PLUS_V1_DROP2_Resnet34 import DeepLabv3_plus
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus


# UNetV2, CamDUNet


dev = "cuda:0"  
device = torch.device(dev) 

def test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm, trMeanR, trMeanG, trMeanB):
    
    
    if modeltype=='UNetV1':
        net = UNetV1(classes=1).to(device)
    elif modeltype=='UNetV2':
        net = UNetV2(classes=1).to(device) 
    elif modeltype=='SegNet':
        net = SegNet(classes=1).to(device) 
    elif modeltype=='DinkNet101':
        net =  DinkNet101(num_classes=1).to(device)        
    elif modeltype=='DeepLabv3_plus':
        net = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device) 
    elif modeltype=='CamDUNet':
       net = CamDUNet().to(device)
    elif modeltype=='R2U_Net':
       net = R2U_Net(img_ch=3,output_ch=1).to(device)        
    elif modeltype=='AttU_Net':
       net = AttU_Net(img_ch=3,output_ch=1).to(device)                
    elif modeltype=='R2AttU_Net':
       net = R2AttU_Net(img_ch=3,output_ch=1).to(device)   
    elif modeltype=='NestedUNet':
       net = NestedUNet(in_ch=3, out_ch=1).to(device)         
    elif modeltype=='DualNorm_Unet':
       net = DualNorm_Unet(n_channels=3, n_classes=1).to(device)       
    elif modeltype=='InceptionUNet':
       net = InceptionUNet(n_channels=3, n_classes=1, bilinear=True).to(device)       
    elif modeltype=='AttU_Net_with_scAG':
       net = AttU_Net_with_scAG(img_ch=3, output_ch=1,ratio=16).to(device)       
    elif modeltype=='FSFNet':
       net = FSFNet(num_classes=1).to(device)             
    elif modeltype=='LMFFNet':
       net = LMFFNet(classes=1, block_1=3, block_2=8) .to(device)
    elif modeltype=='LMFFNet2':
       net = LMFFNet2(classes=1, block_1=3, block_2=8) .to(device)
    elif modeltype=='FASSDNet':
       net = FASSDNet(n_classes=1, alpha=2).to(device) 
    elif modeltype=='ENet':
       net = ENet(classes=1).to(device)  
    elif modeltype=='ELANet':  
        net = ELANet().to(device)              
  
       
 
        
    net.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel{}.pt".format(i))))

    jI = 0
    totalBatches = 0
    test_losses = []
    net.eval()
    with torch.no_grad():
        t_losses = []
        t=0
        for testim, testmas in test_generator:
            images=testim.to(device)
            masks=testmas.to(device)
            outputs = net(images)
            if t==0:
                fig=plt.figure()
                axes=[]
                fimage=images[0].permute(1, 2, 0)
                fimage[:,:,0]=(images[0][0,:,:])
                fimage[:,:,1]=(images[0][1,:,:])
                fimage[:,:,2]=(images[0][2,:,:])
                fimage=fimage.cpu().numpy()
                axes.append(fig.add_subplot(1, 2, 1))
                foutput=outputs[0].permute(1, 2, 0)
                foutput=foutput.cpu().numpy()
                plt.imshow(np.squeeze(foutput, axis=2),  cmap='gray')
                subplot_title=("Test Predicted Mask")
                axes[-1].set_title(subplot_title)
                axes.append(fig.add_subplot(1, 2, 2))
                fmask=masks[0].permute(1, 2, 0)
                fmask=fmask.cpu().numpy()
                plt.imshow(np.squeeze(fmask, axis=2),  cmap='gray')
                subplot_title=("Ground Truth Mask")
                axes[-1].set_title(subplot_title)
                n_curve = 'mask_comparison.png'
                plt.savefig(os.path.join(pathm, n_curve))
                plt.show()
                segplot(pathm, lim, fimage, foutput, fmask,  trMeanR, trMeanG, trMeanB)
            losst=nn.BCEWithLogitsLoss()
            output = losst(outputs, masks)
            t_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
            t+=1
                 
    dn=jI/totalBatches
    dni=dn.item()
    test_loss = np.mean(t_losses)
    test_losses.append(test_loss)
    testFile.write(str(test_losses[0])+"\n")
    testaccFile.write(str(dni)+"\n")
    print("Test Jaccard:",dni)
