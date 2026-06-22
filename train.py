import warnings, os

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

torch.serialization.add_safe_globals([RTDETRDetectionModel])
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    # model.load('') # loading pretrain weights
    # model.load('paper2/val/ECAV3.pt')  # map_location 根据需要改
  
    
    model.train(data='F:/Dpan/RTDETR622new/dataset/Finaldata.yaml',
                cache=True,
                imgsz=640,
                epochs=100,
                batch=4, 
                workers=0, 
                # device='0,1',  
                # resume='', 
                project='runs/train',
                name='starnet-og',
                )
    
