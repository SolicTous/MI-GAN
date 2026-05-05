
import argparse
import os
import warnings
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pickle
import PIL.Image
from time import time
import torch
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("D:/PyTools/")
from ImageChunk import *
from DebugPhoto import *
from OnnxLoad import *
from ChangeImage import *

from MI_GAN.lib.model_zoo.migan_inference import Generator as MIGAN
from MI_GAN.lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)

warnings.filterwarnings("ignore")

from ImageChunk import *
from OnnxLoad import *

def process(img, mpath, input_dict, output, sig, Scale, crop_dis=8):
    TileSize = 256
    chunker_l = ImageChunker(TileSize, TileSize, crop_dis, debug=False)
    chunker_h = ImageChunker(TileSize * Scale, TileSize * Scale, crop_dis * Scale, debug=False)
    if sig > 0:
        img = cv2.bilateralFilter(img, sig, sig*3, sig*3)
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.0
    imgX = cv2.resize(img, (0, 0), fx=Scale, fy=Scale)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    if len(imgX.shape) == 2:
        imgX = np.expand_dims(imgX, axis=2)
    imglist_l = chunker_l.dimension_preprocess(img)
    pics = []

    AIMODEL = ModelLoader(model_filepath=mpath, inputs=input_dict,
                          output=output, gpu_use = True, dml=True)

    op_time = 0
    op_cout = 0

    for img_idx in range(imglist_l[:, 0, 0, 0].shape[0]):
        input_im = imglist_l[img_idx, :, :, :]
        input_im = np.expand_dims(input_im, 0)
        input_dict = {'x:0': input_im.astype(np.float32)}

        output = AIMODEL.test(input_dict)[0]
        out = output[0, :, :, :]

        pics.append(out)

    pics = [np.expand_dims(pic, axis=0) for pic in pics]
    chunked_images = np.concatenate(pics)
    out = chunker_h.dimension_postprocess(chunked_images, imgX) * 255
    out = np.clip(out, 0, 255).astype(np.uint8)
    if out.shape[2]>1:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    else:
        out = out[:, :, 0]
    return out


def resize(image, max_size, inter = cv2.INTER_CUBIC):
    w, h = image.shape[1], image.shape[0]
    if w > max_size or h > max_size:
        resize_ratio = max_size / w if w > h else max_size / h
        image = cv2.resize(image, (int(h * resize_ratio), int(w * resize_ratio)), inter)
    return image


def preprocess(img: Image, mask: Image, resolution: int) -> torch.Tensor:
    img = cv2.resize(img, (resolution, resolution), cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (resolution, resolution), cv2.INTER_NEAREST)
    mask = mask[:, :, np.newaxis] // 255
    img = torch.Tensor(img).float() * 2 / 255 - 1
    mask = torch.Tensor(mask).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    mask = mask.permute(2, 0, 1).unsqueeze(0)
    x = torch.cat([mask - 0.5, img * mask], dim=1)

    # show = ((img*mask+1)/2).cpu().numpy()[0]
    # show = np.transpose(show, (1, 2, 0))
    # show = np.clip(show*255, 0, 255).astype(np.uint8)
    # cv2.imshow('img', cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="One of [migan-256, migan-512, comodgan-256, comodgan-512]",
                        default='migan-512')
    parser.add_argument("--model-path", type=str, help="Saved model path.",
                        default=r'D:\Models\Actual\MIGAN\migan\migan_512_places2.pt')
    parser.add_argument("--images-dir", type=Path, help="Path to images directory.",
                        default=r'D:\Datasets\MATTE\DIS5K\im')
                        # default=r'D:\Download\in')
    parser.add_argument("--masks-dir", type=Path, help="Path to masks directory.",
                        default=r'D:\Datasets\MATTE\DIS5K\gt')
                        # default=r'D:\Download\msk')
    parser.add_argument("--invert-mask", default=True, help="Invert mask? (make 0-known, 1-hole)")
    parser.add_argument("--output-dir", type=Path, help="Output directory.",
                        default=r'C:\Users\marsel\PycharmProjects\MI-GAN\output')
    parser.add_argument("--device", type=str, help="Device.", default="cuda")
    return parser.parse_args()

def check_borders(bbox, init_image):
    for i in range(len(bbox)):
        if bbox[i] < 0:
            bbox[i] = 0
        elif (i == 2) and (bbox[i] >= init_image.shape[1]):
            bbox[i] = init_image.shape[1]
        elif (i == 3) and (bbox[i] >= init_image.shape[0]):
            bbox[i] = init_image.shape[0]
    return bbox


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    RECT_MODE = True

    cuda = False
    if args.device == "cuda":
        cuda = True

    if args.model_name == "migan-256":
        resolution = 256
        model = MIGAN(resolution=256)
    elif args.model_name == "migan-512":
        resolution = 512
        model = MIGAN(resolution=512)
    elif args.model_name == "comodgan-256":
        resolution = 256
        comodgan_mapping = CoModGANMapping(num_ws=14)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    elif args.model_name == "comodgan-512":
        resolution = 512
        comodgan_mapping = CoModGANMapping(num_ws=16)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    else:
        raise Exception("Unsupported model name.")

    model.load_state_dict(torch.load(args.model_path))
    if cuda:
        model = model.to("cuda")
    model.eval()

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(args.images_dir, "**", f"*{img_extension}"), recursive=True)

    img_paths = sorted(img_paths)
    idx = 0

    import onnxruntime as ort
    onnx_path = r"D:\Models\Actual\MIGAN\migan\migan_512_places2.onnx"
    model_onnx = ort.InferenceSession(onnx_path,
                                      # providers=[('DmlExecutionProvider', {'device_id': (0), })])
                                      providers=[('CPUExecutionProvider')])

    for img_path in tqdm(img_paths):
        idx += 1
        if idx % 100 == 0:
            mask_path = os.path.join(args.masks_dir, "".join(os.path.basename(img_path).split('.')[:-1]) + ".png")

            img = cv2.imread(img_path)
            img_init = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[mask < 200] = 0
            mask[mask > 200] = 255

            Regions = []
            npmask = np.asarray(mask)
            contours, _ = cv2.findContours(npmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            npmask = cv2.cvtColor(npmask, cv2.COLOR_GRAY2BGR)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                e_bbox = [x - int(w * 0.25), y - int(h * 0.25), x + int(w * 1.25), y + int(h * 1.25)]
                coords = check_borders(e_bbox, npmask)
                Regions.append(coords)

            for bbox in Regions:
                img_work = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                mask_work = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # расширение маски с небольшим адаптивным ядром
                kern = round(max(mask_work.shape[:2])/512 * 5)
                dilate_kernel = np.ones((kern, kern), np.uint8)
                mask_nb = cv2.dilate(mask_work, dilate_kernel, iterations=5)

                # блюр после половинного сужения (плавный переход м/у исходной границей и расширенной)
                # применяется только к итоговому изображению, не идёт на вход нейросети
                eroded = cv2.erode(mask_nb, np.ones((kern//2, kern//2), np.uint8), iterations=2)
                bc = int(25 * max(mask_work.shape[:2]) / 512) // 3 * 3
                if bc % 2 == 0:
                    bc += 1
                mask_blur = cv2.GaussianBlur(eroded, (bc, bc) if bc != 0 else (3, 3), 0)

                mask_bi = mask_nb.copy()
                mask_bi[mask_bi > 0] = 255
                if args.invert_mask:
                    mask_bi = 255 - mask_bi.copy()


                x = preprocess(img_work.copy(), mask_bi, resolution)
                if cuda:
                    x = x.to("cuda")
                with torch.no_grad():
                    x = x.permute(0, 2, 3, 1)

                    # from convert_onnx import convert_onnx
                    # convert_onnx(model, args.model_path.replace('.pt','.onnx'), 'cuda', x)
                    # sys.exit()

                    x = x.cpu().numpy()
                    if 'f16' in onnx_path:
                        x = x.astype(np.float16)

                    start = time.time()
                    result_image = model_onnx.run(None, {'input': x})[0]
                    print('Inpaint time - ', time.time() - start)

                    result_image = torch.from_numpy(result_image).cuda().permute(2, 0, 1)

                result_image = (result_image * 0.5 + 0.5).clamp(0, 1) * 255
                result_image = result_image.to(torch.uint8).permute(1, 2, 0).detach().to("cpu").numpy()

                SRx4 = process(result_image, r"D:\Models\AiEditor\onnx\sr\PhotoSRx4.onnx",
                                          {'x:0': (None, None, None, 3)}, 'Identity', sig=7, Scale=4)
                SRx4 = cv2.resize(SRx4, dsize=(img_work.shape[1], img_work.shape[0]),
                                             interpolation=cv2.INTER_CUBIC)

                result_image_bic = cv2.resize(result_image, dsize=(img_work.shape[1], img_work.shape[0]),
                                              interpolation=cv2.INTER_CUBIC)


                mask_blur = mask_blur[:, :, np.newaxis] / 255
                if args.invert_mask:
                    mask_blur = 1 - mask_blur

                final_mask = (mask_blur[:,:,0]*255).astype(np.uint8) + mask_work - (255 - mask_nb).astype(np.uint8)
                img_SRx4 = img.copy()

                img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] * mask_blur + result_image_bic * (1 - mask_blur)
                img_SRx4[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = img_SRx4[bbox[1]:bbox[3], bbox[0]:bbox[2], :] * mask_blur + SRx4 * (1 - mask_blur)

                img_SRx4 = text_text(img_SRx4, 'SRx4', 1)

                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = final_mask

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = text_text(img, 'bicubic', 1)
            img_SRx4 = cv2.cvtColor(img_SRx4, cv2.COLOR_RGB2BGR)
            concat1 = np.concatenate((img_init, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), axis=1)
            concat2 = np.concatenate((img, img_SRx4), axis=1)
            concat = np.concatenate((concat1, concat2), axis=0)

            cv2.imwrite(os.path.join(args.output_dir, str(idx)+'.jpg'), concat)

if __name__ == '__main__':
    main()
