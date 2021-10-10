import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadPagesFromPaths
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect_crop(images_paths_list, weights_path):
    # weights, imgsz = opt.weights, opt.img_size
    ## parameters
    device = 'cpu'
    imgsz = 640
    iou_thres = 0.45
    conf_thres = 0.25
    classes = None
    agnostic_nms = False
    augment = False

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights_path, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # image_paths_list = image_path_list

    dataset = LoadPagesFromPaths(images_paths_list=images_paths_list,
                                 img_size=imgsz)
    # dataset = LoadImages(path=source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img
              ) if device.type != 'cpu' else None  # run once
    crops_dict = dict()
    image_count = 0
    for path, img, im0s, vid_cap in dataset:
        # img :  resized image and size controlled by imgsz/opt.img_size
        # im0s or im0 : original image before resize
        image_count += 1
        image_key = f'img_{image_count}'
        crops_dict[image_key] = dict()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS

        pred = non_max_suppression(pred,
                                   conf_thres,
                                   iou_thres,
                                   classes=classes,
                                   agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        temp_dict = dict()

        for i, cuts_vectors in enumerate(pred):  # cuts per image

            p, s, im0 = Path(path), '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(cuts_vectors):
                # Rescale boxes from img_size to im0 size
                cuts_vectors[:, :4] = scale_coords(img.shape[2:],
                                                   cuts_vectors[:, :4],
                                                   im0.shape).round()
                temp_dict['page_path'] = path
                # print(cuts_coords)
                # print(cuts_coords.dtype)
                temp_dict['cuts_vectors'] = cuts_vectors.numpy()
                crops_dict[image_key] = temp_dict

                # Print results
                for c in cuts_vectors[:, -1].unique():
                    n = (cuts_vectors[:,
                         -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

    # print(crops_dict.keys())
    # print(crops_dict.values())
    print(f'Done. ({time.time() - t0:.3f}s)')
    return crops_dict

if __name__ == '__main__':
    pass