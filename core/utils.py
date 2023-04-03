import pickle
import numpy as np
import os
import torch


def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1+padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1+size_bb, y1+size_bb]

def pad_img(bbox, img, padding_ratio=0.2):
    bbox[[0,1]] = np.floor(bbox[[0,1]])
    bbox[[2,3]] = np.ceil(bbox[[2,3]])
    bbox = bbox.astype(np.uint16)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    sidelen = np.floor(max(width, height) * (1 + padding_ratio)).astype(np.uint16)
    padded_img = np.zeros((sidelen, sidelen, 3), dtype=np.uint8)

    x_offset = (sidelen - width) // 2
    y_offset = (sidelen - height) // 2
    px2 = x_offset + bbox[2] - bbox[0]
    py2 = y_offset + bbox[3] - bbox[1]
    #bbox = np.ceil(bbox).astype(np.uint8)
    padded_img[y_offset:py2, x_offset:px2, :] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

    bbox_offset = np.asarray([bbox[0] - x_offset, bbox[1] - y_offset, bbox[2] + x_offset, bbox[3] + y_offset])

    return padded_img, sidelen, bbox, bbox_offset


def mymkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lm_weights(device):
    w = torch.ones(68).to(device)
    w[28:31] = 10
    w[48:68] = 10
    norm_w = w / w.sum()
    return norm_w


def save_obj(path, v, f, c):
    with open(path, 'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n' %
                       (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()
