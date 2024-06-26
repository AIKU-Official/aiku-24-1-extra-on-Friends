import argparse
import cv2
import torch
import time
import os

from utils.inference.image_processing import crop_face, get_final_image, show_images
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, face_enhancement
from utils.inference.core import model_inference
from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_path', type=str, help='path to the source image')
    parser.add_argument('--target_path', type=str, help='path to the target images')
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    # Initialize models
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    # main model for generation
    G = AEI_Net(backbone='unet', num_blocks=2, c_id=512)
    G.eval()
    G.load_state_dict(torch.load('weights/G_unet_2blocks.pth', map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    use_sr = True
    if use_sr:
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        #opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
 
        source_full = cv2.imread(args.source_path)
        crop_size = 224 # don't change this !!

        # check, if we can detect face on the source image
        try:
            source = crop_face(source_full, app, crop_size)[0]
            source = [source[:, :, ::-1]]
            print("Everything is ok!")
        except TypeError:
            print("Bad source images")

        target_full = cv2.imread(args.target_path)
        full_frames = [target_full]
        target = get_target(full_frames, app, crop_size)

        final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
        full_frames, source, target, netArc, G, app, set_target = False, crop_size=crop_size, BS=4)

        if use_sr: final_frames_list = face_enhancement(final_frames_list, model)
        result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
        cv2.imwrite(args.save_path, result)
        print("saved!")