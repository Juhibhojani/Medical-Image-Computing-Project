from blocks import *
from DFLT_module import DFLT
from timm.models.registry import register_model

class StudentModel(nn.Module):
    """
        Args:
        image_size (int): Size of input images. Default: 224
        in_channel (int): Number of channels in the input images. Default: 3
        num_blocks (List(int)): Number of blocks at each stage. Default: [2, 2, 3, 3]
        channels (List(int)): The output dimension at each stage. Default: [64, 96, 192, 256]
        patch_size (tuple): The patch size used for the transformer (DFLT block). Default: (2,2)
        num_classes (int): Number of classes in the dataset. Default: 7 (for HAM-10000 dataset).
        use_distillation (bool): A boolean to indicate if distillation will be used or not. Default: True (for HDKD).
        multi_distill (bool) : A boolean to indicate if more distillation tokens are required or not for distillation
        heads (int): Number of heads in the DFLT block. Default: 8
        expansion (int): Expansion ratio. Default: 4
        dim_head (int): Dimension per head. Default: 32
        dropout (int): Dropout rate. Default 0

    """
    def __init__(self, image_size, in_channel, num_blocks, channels,patch_size,num_classes=7,use_distillation=True, multi_distill=False,heads=8, expansion=4, dim_head = 32, dropout = 0., emb_dropout = 0., **kwargs):
        super().__init__()
        # size of image taken
        ih,iw= image_size
        # arg to use distillation or not
        self.use_distillation = use_distillation
        self.multi_distill = multi_distill
        block={'stem':conv_3x3_bn,'MBConv':MBConv,'MBCSA':MBCSA}
        # setting up same architecture as teacher but with different number of parameters
        self.stage1 = self._make_block(block['stem'],in_channel,channels[0],3,num_blocks[0])
        self.stage2 = self._make_block(block['MBCSA'],channels[0],channels[1],3,num_blocks[1])
        self.stage3 = self._make_block(block['MBCSA'],channels[1],channels[2],3,num_blocks[2])
        # downsampling at each stage, thats why this output
        feature_resolution = (ih//8 , iw//8)

        self.DFLT_module = self._make_DFLT_module(feature_resolution,patch_size,channels[3],num_blocks[3],heads,expansion,channels[2],use_distillation,multi_distill,dim_head,dropout,emb_dropout)
        # convert from cls token dim to number of classes via linear layer
        self.head_cls = nn.Linear(channels[3],num_classes)
        
        if self.use_distillation:
            self.head_distill = nn.Linear(channels[3], num_classes)

        if self.use_distillation and self.multi_distill:
            # projection heads for distillation
            self.proj_stage2 = nn.Linear(channels[3], channels[3])
            self.proj_stage3 = nn.Linear(channels[3], channels[3])


    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if self.use_distillation:
            if self.multi_distill:
                cls_token, d2, d3, dlogit = self.DFLT_module(x)

                # classification outputs
                cls_out = self.head_cls(cls_token)
                logit_out = self.head_distill(dlogit)

                # projection for feature alignment
                d2 = self.proj_stage2(d2)
                d3 = self.proj_stage3(d3)

                return cls_out, d2, d3, logit_out
            else:
                cls_token, distill_token = self.DFLT_module(x)

                cls_out = self.head_cls(cls_token)
                distill_out = self.head_distill(distill_token)

                return cls_out, distill_out
        else:
            x_cls = self.DFLT_module(x)
            x_cls = self.head_cls(x_cls)
            return x_cls

    def _make_block(self,block,inp,oup,kernel_size,depth):
        layers=nn.ModuleList([])
        for i in range(depth):
            if i==0:
                layers.append(block(inp,oup,kernel_size,downsample=True))
            else:
                layers.append(block(oup,oup,kernel_size))

        return nn.Sequential(*layers)


    def _make_DFLT_module(self,feature_resolution,patch_size,dim,depth,heads,expansion,channels,use_distillation,multi_distill,dim_head,dropout,emb_dropout):
        return DFLT(feature_resolution,patch_size,dim,depth,heads,expansion,channels,use_distillation,multi_distill,dim_head,dropout,emb_dropout)


@register_model
def student_model(**kwargs):
    # changing student model to have more layers, i.e. essentially training a teacher model with transformer to see what happens with full data setting
    num_blocks = [2, 2, 3, 3]
    channels = [64, 96, 192,256]
    # print(f"Creating model with num_blocks:{num_blocks[0],num_blocks[1],num_blocks[2],num_blocks[3]}")
    return StudentModel((224, 224), 3, num_blocks, channels, patch_size=(2,2), use_distillation=False, **kwargs)

@register_model
def HDKD(**kwargs):
    num_blocks = [2, 2, 3, 3]
    channels = [64, 96, 192,256]
    print(f"It's multiple distill tokens setup!")
    return StudentModel((224, 224), 3, num_blocks, channels, patch_size=(2,2), use_distillation=True,multi_distill=True, **kwargs)
