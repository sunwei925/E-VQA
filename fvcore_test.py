import torch
import models.UIQA as UIQA

from fvcore.nn import FlopCountAnalysis 

# model = baseline_Swin_motion.Model_SwinT_LSVQ()
# model = baseline_Swin_motion.Model_MobileNet_V2_LSVQ()

model = UIQA.UVQA_Model()
inputs = torch.randn(1,4,3,384,384)
flops = FlopCountAnalysis(model, inputs=(inputs,inputs))
print('FLOPS:', flops.total()/1e9, 'GFLOPs')
