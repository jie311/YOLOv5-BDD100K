import torch
from common import Conv


def load_weights(model, pretrain, cut_index=None, training=False):
    # 模型推理
    if not training:
        for m in model.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                conv = m.conv
                conv.bias = torch.nn.Parameter(torch.zeros(conv.weight.size(0), device=conv.weight.device))
                delattr(m, 'bn')
                m.forward = m.forward_fuse

    model_dict = model.state_dict()
    pret_dict = pretrain.state_dict()
    model_list = [[k, v] for k, v in model_dict.items()]
    pret_list = [[k, v] for k, v in pret_dict.items()]

    assert len(model_list) == len(pret_list)

    cut_index = len(model_list) if cut_index is None else cut_index
    for i in range(len(model_list)):
        if model_list[i][0] == pret_list[i][0] and i < cut_index:
            model_list[i][1] = pret_list[i][1]
        else:
            print('layers miss, model layer: %s,  pret layer: %s' % (model_list[i][0], pret_list[i][0]))

    for k, v in model_list:
        model_dict[k] = v

    model.load_state_dict(model_dict)


def fuse_conv_bn(m):
    conv, bn = m.conv, m.bn

    w = conv.weight.view(conv.out_channels, -1)
    b = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias

    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    w_new = torch.mm(torch.diag(gamma.div(torch.sqrt(eps + var))), w).view(conv.weight.shape)
    b_new = gamma.div(torch.sqrt(eps + var)) * (b - mean) + beta

    m.conv.requires_grad_(False)
    m.conv.weight.copy_(w_new)
    m.conv.bias.copy_(b_new)

    delattr(m, 'bn')


# def load_model(name, anchors, strides, num_cls, pretrain_path, cut_index, half, device, training):
#     if name == 'yolo5':
#         model = YOLO5(anchors, strides, num_cls, device)
#
#     elif name == 'yolop':
#         model = YOLOP(anchors, strides, num_cls, device)
#
#     else:
#         raise RuntimeError('model_type does not exist')
#
#     pretrain = torch.hub.load('ultralytics/yolov5', 'custom', pretrain_path)
#
#     load_weights(model, pretrain, cut_index, training)
#
#     model.eval()
#
#     half = half & (device != 'cpu')
#
#     return model.to(torch.float16) if half else model.to(torch.float32)

def save_model(model):
    pass
