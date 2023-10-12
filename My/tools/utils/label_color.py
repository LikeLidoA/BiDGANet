import numpy as np
import scipy.io as sio
from PIL import Image
import torch
import os

label_color_city = [(128, 64, 128), (244, 35, 232), (70, 70, 70)
                    # 0 = road, 1 = sidewalk, 2 = building
    , (102, 102, 156), (190, 153, 153), (153, 153, 153)
                    # 3 = wall, 4 = fence, 5 = pole
    , (250, 170, 30), (220, 220, 0), (107, 142, 35)
                    # 6 = traffic light, 7 = traffic sign, 8 = vegetation
    , (152, 251, 152), (70, 130, 180), (220, 20, 60)
                    # 9 = terrain, 10 = sky, 11 = person
    , (255, 0, 0), (0, 0, 142), (0, 0, 70)
                    # 12 = rider, 13 = car, 14 = truck
    , (0, 60, 100), (0, 80, 100), (0, 0, 230)
                    # 15 = bus, 16 = train, 17 = motocycle
    , (119, 11, 32), (0, 0, 0)]
# 18 = bicycle     #背景

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
# Road_marking = [255, 69, 0]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

label_color_voc = np.array([  # 21(加背景)
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
    [0, 0, 0]])  # 背景

matfn = './color150.mat'


def get_color_list(dataset):
    if dataset == "ade20k":
        mat = sio.loadmat(matfn)
        color_table = mat['colors']
        shape = color_table.shape
        # print(shape) #(150, 3)
        color_list = [list(color_table[i]) for i in range(shape[0])]
        color_list = np.array(color_list)
    elif dataset == "city":
        color_list = [list(label_color_city[i]) for i in range(20)]
        color_list = np.array(color_list)
    elif dataset == "camvid":
        color_list = np.array(
            [Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    else:  # VOC2012
        color_list = label_color_voc
    return color_list


def read_labelcolors(image, dataset):  # 传入的是图片数组

    color_list = get_color_list(dataset)
    r = image.copy()
    g = image.copy()
    b = image.copy()

    for color in range(0, color_list.shape[0]):  # -1可去掉背景
        r[image == color] = color_list[color, 0]
        g[image == color] = color_list[color, 1]
        b[image == color] = color_list[color, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    return im


def onehot_to_rgb(onehot, dataset):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            dataset - dataset name
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    color_list = get_color_list(dataset)
    for k, color in enumerate(color_list):
        output[single_layer == k] = color
    return np.uint8(output)


def get_class_weight(dataset):
    if dataset == "city":
        n_classes = len(label_color_city)
        class_weights = torch.FloatTensor([2.8149201869965, 6.9850029945374, 3.7890393733978,
                                           9.9428062438965, 9.7702074050903, 9.5110931396484,
                                           10.311357498169, 10.026463508606, 4.6323022842407,
                                           9.5608062744141, 7.8698215484619, 9.5168733596802,
                                           10.373730659485, 6.6616044044495, 10.260489463806,
                                           10.287888526917, 10.289801597595, 10.405355453491,
                                           10.138095855713, 0.0
                                           ]).cuda()
        return class_weights
    elif dataset == "camvid":
        class_weights = torch.FloatTensor([0.2595, 0.1826, 4.5640, 0.9051,  # 0.1417
                                           0.3826, 9.6446, 1.8418, 0.6823, 6.2478,
                                           7.3614, 1.0974, 0.0]).cuda()

        return class_weights
