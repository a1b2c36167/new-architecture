import os

import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i >= 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        create_grids(self, 32, 1, device=device)

    def forward(self, p, img_size, var=None):
        bs, nG = p.shape[0], p.shape[-1]
        if self.img_size != img_size:
            create_grids(self, img_size, nG, p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # inference
        io = p.clone()  # inference output
        io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
        io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
        # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
        io[..., :4] *= self.stride
        if self.nC == 1:
            io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

        # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
        return io.view(bs, -1, 5 + self.nC), p


def create_grids(self, img_size, nG, device='cpu'):
    self.img_size = img_size
    self.stride = img_size / nG

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2).to(device)
    self.nG = torch.FloatTensor([nG]).to(device)


class Darknet(nn.Module):
    def __init__(self, cfg_path=None, mode=None, img_size=416):
        super(Darknet, self).__init__()

        self.mode = mode
        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        #self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = get_yolo_layers(self)

    def forward(self, x, get_feature_map=None, var=None):
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)
            #if self.mode == 2 :
            #    print("a", i, mtype)
            #    if type(layer_outputs[i]) is torch.Tensor:
            #        print(layer_outputs[i].size())

        # inference output, training output
        io, p = list(zip(*output))
        if get_feature_map:
            feature_map = layer_outputs[get_feature_map]
            if self.mode == 1:
                f1 = F.interpolate(layer_outputs[79], size=[52, 52])
                f2 = F.interpolate(layer_outputs[91], size=[52, 52])
                f3 = layer_outputs[103]
                feature_map = torch.cat((f1,f2,f3),1)
            elif self.mode == 2:
                f1 = F.interpolate(layer_outputs[6], size=[26, 26])
                f2 = layer_outputs[13]
                feature_map = torch.cat((f1,f2),1)

            return torch.cat(io, 1), p, feature_map
        else:
            return torch.cat(io, 1), p


def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


class Darknet53(nn.Module):
    def __init__(self, cin, num_classes):
        super(Darknet53, self).__init__()

        self.cin = cin
        self.num_classes = num_classes

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, self.num_classes)

    def forward(self, x):
        #print("1",x.size())
        out = self.global_avg_pool(x)
        #print("2",out.size())
        out = out.view(-1, 768)
        #print("3",out.size())
        out = self.fc(out)
        #print("4",out.size())

        return out


class DarknetPlus(nn.Module):
    def __init__(self, cfg_path=None, cfg_path2=None, img_size=416):
        super(DarknetPlus, self).__init__()
        self.img_size = img_size
        self.img_size2 = img_size // 16
        self.img_size3 = 8

        self.model1 = Darknet(cfg_path, 1, self.img_size)
        self.model2 = Darknet(cfg_path2, 2, self.img_size2)
        self.model3 = Darknet53(768, 3)

    def forward(self, img, targets=None, LS_targets=None, conf_thres1=0.5, nms_thres1=0.5, conf_thres2=0.5, nms_thres2=0.5, var=None):
        # for training or testing
        if targets is not None and LS_targets is not None:
            inf_out1, train_out1, inf_out2, train_out2, inf_out3 = self.forward_train(img, targets, LS_targets)
            if self.training:  # for training
                #print(len(train_out1),len(train_out2),len(inf_out3))
                return train_out1, train_out2, inf_out3
            else:              # for testing
                return inf_out1, train_out1, inf_out2, train_out2, inf_out3
        # for inferencing
        elif targets is None and LS_targets is None:
            det1, det2 = self.forward_inference(img, conf_thres1, nms_thres1, conf_thres2, nms_thres2)
            return det1, det2
        else:
            print('targets: ', targets is None)
            print('LS_targetsis: ', LS_targetsis is None)
            print('error')

    def forward_train(self, img, targets=None, LS_targets=None):
        # Run model1
        inf_out1, train_out1, feature_map1 = self.model1(img, get_feature_map=103)
        self.nG = feature_map1.shape[-1]
        self.img_size2 = self.nG // 2
        # generate crop_xyxy
        crop_xyxy = targets.clone()
        x, y = targets[:, 2], targets[:, 3]
        w, h = targets[:, 4], targets[:, 5]
        crop_xyxy[:, 2], crop_xyxy[:, 3] = x - (w / 2), y - (h / 2)
        crop_xyxy[:, 4], crop_xyxy[:, 5] = x + (w / 2), y + (h / 2)
        crop_xyxy[:, 2:] *= self.nG

        # limit crop_xyxy
        crop_xyxy = self.xyxy_limit(crop_xyxy)

        # Run model2
        feature_map1 = self.crop_roi(feature_map1, crop_xyxy, self.img_size2)
        inf_out2, train_out2, feature_map2 = self.model2(feature_map1, get_feature_map=13)
        self.nG2 = feature_map2.shape[-1]

        # generate crop_xyxy
        LS_c_nt = len(LS_targets[LS_targets[:, 1] > 2, 1])
        if LS_c_nt == 0:  # no need to run model3 (no direction light)
            return inf_out1, train_out1, inf_out2, train_out2, None
        else:             # need to run model3 (has direction light)
            crop_xyxy = LS_targets[LS_targets[:, 1] > 2, :].clone()
            x, y = LS_targets[LS_targets[:, 1] > 2, 2], LS_targets[LS_targets[:, 1] > 2, 3]
            w, h = LS_targets[LS_targets[:, 1] > 2, 4], LS_targets[LS_targets[:, 1] > 2, 5]
            crop_xyxy[:, 2], crop_xyxy[:, 3] = x - (w / 2), y - (h / 2)
            crop_xyxy[:, 4], crop_xyxy[:, 5] = x + (w / 2), y + (h / 2)
            crop_xyxy[:, 2:] *= self.nG2
        # limit crop_xyxy
        crop_xyxy = self.xyxy_limit(crop_xyxy)

        # Run model3
        #print("i",feature_map2.size())
        feature_map2 = self.crop_roi(feature_map2, crop_xyxy, self.img_size3)
        #print("ii",feature_map2.size())
        inf_out3 = self.model3(feature_map2)
        return inf_out1, train_out1, inf_out2, train_out2, inf_out3

    def forward_inference(self, img, conf_thres1=0.5, nms_thres1=0.5, conf_thres2=0.5, nms_thres2=0.5):
        # Run model1
        inf_out1, train_out1, feature_map1 = self.model1(img, get_feature_map=103)
        self.nG = feature_map1.shape[-1]
        self.stride = 8
        self.img_size2 = self.nG // 2

        # generate crop_xyxy
        det1 = non_max_suppression(inf_out1, conf_thres1, nms_thres1)
        crop_xyxy = []
        for i, det in enumerate(det1):
            if det is not None:
                image = i * torch.ones(len(det), 1).to(det.device)
                cls = det[:, 6].clone().unsqueeze(dim=1)
                xyxy = det[:, 0: 4].clone() / self.stride
                crop_xyxy.append(torch.cat((image, cls, xyxy), 1))

        # need or not need to run model2
        if len(crop_xyxy):  # has light
            crop_xyxy = torch.cat(crop_xyxy, 0)
        else:               # no light
            return det1, None

        # limit crop_xyxy
        crop_xyxy = self.xyxy_limit(crop_xyxy)

        # Run model2
        feature_map1 = self.crop_roi(feature_map1, crop_xyxy, self.img_size2)
        inf_out2, train_out2, feature_map2 = self.model2(feature_map1, get_feature_map=13)
        self.nG2 = feature_map2.shape[-1]

        # generate crop_xyxy
        det2 = non_max_suppression(inf_out2, conf_thres2, nms_thres2)
        crop_xyxy = []
        for i, det in enumerate(det2):
            if det is not None:
                image = i * torch.ones(len(det), 1).to(det.device)
                cls = det[:, 6].clone().unsqueeze(dim=1)
                xyxy = det[:, 0: 4].clone()
                crop_xyxy.append(torch.cat((image, cls, xyxy), 1))

        # need or not need to run model2
        if len(crop_xyxy) == 0:       # no direction light
            return det1, det2
        else:                         # has direction light
            crop_xyxy = torch.cat(crop_xyxy, 0)
            crop_xyxy = crop_xyxy[crop_xyxy[:, 1]==3, :]

            # transform pred2's coordinate
            det2 = self.transform_pred2_coord(det1, det2)
            if len(crop_xyxy) == 0:   # no direction light
                return det1, det2

        # limit crop_xyxy
        crop_xyxy = self.xyxy_limit(crop_xyxy)

        # Run model3
        feature_map2 = self.crop_roi(feature_map2, crop_xyxy, self.img_size3)
        inf_out3 = self.model3(feature_map2)

        # get det3
        _, det3 = torch.max(F.softmax(inf_out3, dim=1), 1)
        det3 = det3.to(device=inf_out3.device, dtype=inf_out3.dtype)

        # transform pred2's direction classes
        det2 = self.transform_pred2_class(det2, det3)
        return det1, det2

    def xyxy_limit(self, crop_xyxy):
        crop_xyxy = crop_xyxy.round()
        crop_xyxy[crop_xyxy[:, 2] == self.nG, 2] -= 1
        crop_xyxy[crop_xyxy[:, 3] == self.nG, 3] -= 1
        crop_xyxy[crop_xyxy[:, 2] == crop_xyxy[:, 4], 4] += 1
        crop_xyxy[crop_xyxy[:, 3] == crop_xyxy[:, 5], 5] += 1

        crop_xyxy[crop_xyxy[:, 2] < 0, 2] = 0
        crop_xyxy[crop_xyxy[:, 3] < 0, 3] = 0
        crop_xyxy[crop_xyxy[:, 4] > self.nG, 4] = self.nG - 1
        crop_xyxy[crop_xyxy[:, 5] > self.nG, 5] = self.nG- 1
        crop_xyxy = crop_xyxy.cpu().numpy().astype(int)
        return crop_xyxy

    def crop_roi(self, feature_map, crop_xyxy, img_size):
        new_feature_map = []
        for i, image in enumerate(crop_xyxy[:, 0]):
            crop_feature = feature_map[image, :, crop_xyxy[i, 3]: crop_xyxy[i, 5], crop_xyxy[i, 2]: crop_xyxy[i, 4]].clone()  # b, c, h, w
            crop_feature = crop_feature.unsqueeze(dim=0)
            crop_feature = F.interpolate(crop_feature, size=(img_size, img_size), mode='nearest')
            new_feature_map.append(crop_feature)
        new_feature_map = torch.cat(new_feature_map, 0)
        return new_feature_map

    def transform_GT_with_targets(self, targets):
        LS_targets = targets[targets[:, 1] != 0]
        LS_targets[:, 1] -= 1
        targets = targets[targets[:, 1] == 0]

        LS_classify_targets = LS_targets[LS_targets[:, 1] > 2, 0:2].clone()
        LS_classify_targets[:, 1] -= 3
        LS_targets[LS_targets[:, 1] > 2, 1] = 3

        for i, image in enumerate(targets[:, 0]):
            x = targets[i, 2] - targets[i, 4]/2
            y = targets[i, 3] - targets[i, 5]/2
            w = targets[i, 4]
            h = targets[i, 5]

            j = torch.ones(len(LS_targets)).to(LS_targets.device)
            j = LS_targets[:, 0] == j * image
            LS_targets[j, 2] = (LS_targets[j, 2] - x) / w
            LS_targets[j, 3] = (LS_targets[j, 3] - y) / h
            LS_targets[j, 4] = LS_targets[j, 4] / w
            LS_targets[j, 5] = LS_targets[j, 5] / h
        return targets, LS_targets, LS_classify_targets

    def transform_pred2_coord(self, det1, det2):
        for det in det2:
            if det is not None:
                i = det[:, 0] < 0
                det[i, 0] = 0
                i = det[:, 1] < 0
                det[i, 1] = 0
                i = det[:, 2] >= self.img_size2
                det[i, 2] = self.img_size2 - 1
                i = det[:, 3] >= self.img_size2
                det[i, 3] = self.img_size2 - 1

        index = -1
        for det in det1:
            if det is None:
                continue
            for obj in det:
                index += 1
                if det2[index] is None:
                    continue
                    
                det2[index][:, :4] /= self.img_size2
                det2[index][:, 0] *= (obj[2] - obj[0])
                det2[index][:, 1] *= (obj[3] - obj[1])
                det2[index][:, 2] *= (obj[2] - obj[0])
                det2[index][:, 3] *= (obj[3] - obj[1])

                det2[index][:, 0] += obj[0]
                det2[index][:, 1] += obj[1]
                det2[index][:, 2] += obj[0]
                det2[index][:, 3] += obj[1]
                det2[index][:, 6] += 1
        return det2

    def transform_pred2_class(self, det2, det3):
        det3 += 4
        temp1 = 0
        for det in det2:
            if det is None:
                continue

            i = det[:, 6]==4
            temp2 = sum(i.cpu().numpy())
            det[i, 6] = det3[temp1: temp1+temp2]
            temp1 += temp2
        return det2


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found.\nTry https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
