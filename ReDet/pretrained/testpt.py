import torch
from mmcv.runner import load_checkpoint


pth_now_path = r'./re_resnet50_c8_batch256-25b16846.pth'
pth_previous_path = r'./re_resnet50_c8_pretrained.pth'

pth_now = torch.load(pth_now_path)
pth_previous = torch.load(pth_previous_path)

for key, value in pth_now['state_dict'].items():
    assert (pth_now['state_dict'][key] == pth_now['state_dict'][key]).all(), '{}不一致'.format(key)
print('11')

