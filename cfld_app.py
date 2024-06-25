
import torch
import argparse
import copy
import os
from torchvision import transforms
from PIL import Image
import numpy as np

from diffusers import DDPMScheduler
from defaults import pose_transfer_C as cfg
from pose_transfer_train import build_model
from models import UNet, VariationalAutoencoder
from pose_utils import (cords_to_map, draw_pose_from_cords,
                        load_pose_cords_from_strings)


device = 'cuda:0'
default_show_size = ((int(512*0.75), 512))

def build_pose_img(annotation_file, img_path):
    string = annotation_file.loc[os.path.basename(img_path)]
    array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
    pose_map = torch.tensor(cords_to_map(array, (256, 256), (256, 176)).transpose(2, 0, 1), dtype=torch.float32)
    pose_img = torch.tensor(draw_pose_from_cords(array, (256, 256), (256, 176)).transpose(2, 0, 1) / 255., dtype=torch.float32)
    pose_img = torch.cat([pose_img, pose_map], dim=0)
    return pose_img

def build_pose_img_by_keypoints(keypoints_y, keypoints_x):
    array = load_pose_cords_from_strings(keypoints_y, keypoints_x)
    pose_map = torch.tensor(cords_to_map(array, (256, 256), (256, 176)).transpose(2, 0, 1), dtype=torch.float32)
    pose_img = torch.tensor(draw_pose_from_cords(array, (256, 256), (256, 176)).transpose(2, 0, 1) / 255., dtype=torch.float32)
    pose_img = torch.cat([pose_img, pose_map], dim=0)
    return pose_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFLD Test")
    parser.add_argument('--source_path', type=str, help='path to the source image')
    parser.add_argument('--keypoints_path')
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    # 파일 경로에서 첫 번째 줄과 두 번째 줄을 읽어와 변수에 저장
    with open(args.keypoints_path, 'r') as file:
        keypoints_y = str(file.readline().strip())
        keypoints_x = str(file.readline().strip())

    # change pose keypoints
    #keypoints_y = " [30, 55, 54, 90, 126, 54, 86, 121, 118, 163, 200, 117, 162, 220, 31, 32, 34, 35] "
    #keypoints_x = ' [86, 86, 73, 68, 80, 98, 109, 109, 84, 91, 111, 100, 86, 88, 84, 88, 80, 92] '


    pose_img_tensor = build_pose_img_by_keypoints(keypoints_y, keypoints_x).unsqueeze(0)


    noise_scheduler = DDPMScheduler.from_pretrained("pretrained_models/scheduler/scheduler_config.json")
    vae = VariationalAutoencoder(pretrained_path="pretrained_models/vae").eval().requires_grad_(False).to(device)
    model = build_model(cfg).eval().requires_grad_(False).to(device)
    unet = UNet(cfg).eval().requires_grad_(False).to(device)

    # build model
    print(model.load_state_dict(torch.load(os.path.join("checkpoints", "pytorch_model.bin"), map_location="cpu"), strict=False))
    print(unet.load_state_dict(torch.load(os.path.join("checkpoints", "pytorch_model_1.bin"), map_location="cpu"), strict=False))

    # change your image path
    img_from = Image.open(args.source_path).convert("RGB")
    img_from.resize(default_show_size)

    trans = transforms.Compose([
        transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_from_tensor = trans(img_from).unsqueeze(0)


    with torch.no_grad():
        c_new, down_block_additional_residuals, up_block_additional_residuals = model({
            "img_cond": img_from_tensor.to(device), "pose_img": pose_img_tensor.to(device)})
        noisy_latents = torch.randn((1, 4, 64, 64)).to(device)
        weight_dtype = torch.float32
        bsz = 1

        c_new = torch.cat([c_new[:bsz], c_new[:bsz], c_new[bsz:]])
        down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample, sample]).to(dtype=weight_dtype) \
                                            for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), torch.zeros_like(v), v]).to(dtype=weight_dtype) \
                                            for k, v in up_block_additional_residuals.items()}

        noise_scheduler.set_timesteps(cfg.TEST.NUM_INFERENCE_STEPS)
        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_uc, noise_pred_down, noise_pred_full = noise_pred.chunk(3)
            noise_pred = noise_pred_uc + \
                            cfg.TEST.DOWN_BLOCK_GUIDANCE_SCALE * (noise_pred_down - noise_pred_uc) + \
                            cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_down)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

        sampling_imgs = vae.decode(noisy_latents) * 0.5 + 0.5 # denormalize
        sampling_imgs = sampling_imgs.clamp(0, 1)

        image_np = (sampling_imgs[0] * 255.).permute((1, 2, 0)).long().cpu().numpy().astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        image_pil_resized = image_pil.resize(default_show_size)
        image_pil_resized.save(args.save_path)