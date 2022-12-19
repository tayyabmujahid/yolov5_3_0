import argparse
import time
from pathlib import Path
import torch
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadPagesFromPaths
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized
import os


class DetectCrops:
    def __init__(self, images_src_dir, images_dst_dir, wghts_path):

        self.images_src_dir = images_src_dir
        self.images_dest_dir = images_dst_dir
        self.weights_path = wghts_path
        print(self.images_src_dir)
        print(self.images_dest_dir)

    def write_label_file(self, crops_dict):
        print(crops_dict)
        for img_name, temp_dict in crops_dict.items():
            crp = 0
            if temp_dict:
                img_path = temp_dict['page_path']

                cut_vectors = temp_dict['cuts_vectors']

                for cv in cut_vectors:
                    crop_tuple = cv[:4]
                    print(crop_tuple)
                input('i')

    def start_labelling(self):
        print('Labelling')
        images_paths = self.get_image_paths()
        crops_dict = self.detect_crop \
            (images_paths, self.weights_path)

        self.write_label_file(crops_dict)
        self.save_crops(crops_dict)

    def save_crops(self, crops_dict: dict):
        for img_name, temp_dict in crops_dict.items():
            crp = 0
            if temp_dict:
                img_path = temp_dict['page_path']
                cut_vectors = temp_dict['cuts_vectors']
                with Image.open(img_path) as im:
                    for cv in cut_vectors:
                        crop_tuple = cv[:4]
                        if 'price_box' in cv[-1]:
                            crop = im.crop(crop_tuple)
                            crp += 1
                            dest_image_path = os.path.join(self.images_dest_dir, f'{crp}_{img_name}')
                            crop.save(dest_image_path, "PNG")

    def get_image_paths(self) -> list:
        images = os.listdir(self.images_src_dir)
        images_paths = [os.path.join(self.images_src_dir, i) for i in images]
        print(f'Cropping {len(images_paths)} images')
        return images_paths

    def detect_crop(self, images_paths_list, weights_path) -> dict:
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
            image_name = path.split('/')[-1]
            image_key = f'{image_name}'
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
                    cuts_vectors_np = cuts_vectors.numpy().astype(object)
                    for cli, cl in enumerate(cuts_vectors_np):
                        label = names[int(cl[-1])]
                        cuts_vectors_np[cli, -1] = label

                    temp_dict['cuts_vectors'] = cuts_vectors_np
                    temp_dict['img_size'] = list(im0s.shape[:2])
                    crops_dict[image_key] = temp_dict

                    # Print results
                    for c in cuts_vectors[:, -1].unique():
                        n = (cuts_vectors[:,
                             -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
        print(f'Done. ({time.time() - t0:.3f}s)')
        return crops_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--imgs_path', type=str, default='data/images', help='source')
    parser.add_argument('--imgd_path', type=str, default='data/images', help='crops destination')
    opt = parser.parse_args()
    weights_path = opt.weights_path
    imgs_path = opt.imgs_path
    imgd_path = opt.imgd_path

    dc = DetectCrops(imgs_path, imgd_path, weights_path)
    dc.start_labelling()
