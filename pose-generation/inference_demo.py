import os

import cv2
from PIL import Image
import torch
from torchvision import transforms

from config import Config
from model import PoseClassifier, VariationalAutoencoder
from train_vae import generate_pose
from vis_pose import vis_pose

image = None
copied_image = None
target_point = (0, 0)

def mouse_click(event, x, y, flags, param):
    global copied_image
    global target_point
    if event == cv2.EVENT_FLAG_LBUTTON:
        target_point = (x, y)
        cv2.circle(copied_image, (x, y), 10, (0, 0, 255), -1)
        cv2.imshow('image', copied_image)
        copied_image = image.copy()

class InferenceModule(object):
    def __init__(self, cfg, classifier, vae):
        self.classifier = classifier
        self.vae = vae

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cfg = cfg

        self.cluster_keypoints_list = []
        with open(os.path.join('./affordance_data', 'centers_30.txt'), 'r') as f:
            cluster_data_list = list(f.readlines())
        for cluster_data in cluster_data_list:
            cluster_data = cluster_data.split(' ')[:-1]
            cluster_data = [float(x) for x in cluster_data]
            cluster_keypoints = []
            for i in range(0, len(cluster_data), 2):
                cluster_keypoints.append((cluster_data[i], cluster_data[i+1]))
            cluster_keypoints = cluster_keypoints[:-1]
            self.cluster_keypoints_list.append(torch.tensor(cluster_keypoints))
        self.cluster_keypoints_list = torch.stack(self.cluster_keypoints_list)
    
    def inference(self, image, target_point):

        width, height = image.size
        crop_box = ((target_point[0] - (height // 2)), (target_point[1] - (height // 2)), (target_point[0] + (height // 2)), (target_point[1] + (height // 2)))
        zoom_box = ((target_point[0] - (height // 4)), (target_point[1] - (height // 4)), (target_point[0] + (height // 4)), (target_point[1] + (height // 4)))
        image_crop = image.crop(crop_box)
        image_zoom = image.crop(zoom_box)

        image_tensor = self.transform(image).unsqueeze(0)
        image_crop_tensor = self.transform(image_crop).unsqueeze(0)
        image_zoom_tensor = self.transform(image_zoom).unsqueeze(0)

        pose_label = self.classifier(image_tensor, image_crop_tensor, image_zoom_tensor)[0]
        pose_index = torch.argmax(pose_label)
        base_pose = self.cluster_keypoints_list[pose_index]
        one_hot_pose = [0.0] * pose_label.shape[0]
        one_hot_pose[pose_index] = 1.0
        one_hot_pose = torch.tensor(one_hot_pose).unsqueeze(0)

        latent_vector = torch.randn((1, self.cfg.latent_dim))

        sclae_deformation = self.vae.decoder(latent_vector, one_hot_pose, image_tensor, image_crop_tensor, image_zoom_tensor)
        scale = sclae_deformation[:, :2]
        deformation = sclae_deformation[:, 2:]

        pose = generate_pose(base_pose.unsqueeze(0), scale, deformation, torch.tensor(target_point).unsqueeze(0))
        return pose

def inference_demo(image_path):
    global image
    global copied_image
    image = cv2.imread(image_path)
    copied_image = image.copy()
    
    cv2.imshow('image', copied_image)
    cv2.setMouseCallback('image', mouse_click)

    while True:
        k = cv2.waitKey(0)
        if k == 13: # press ENTER
            cv2.destroyAllWindows()
            break
    print(target_point)

    cfg = Config()
    classifier = PoseClassifier(cfg)
    vae = VariationalAutoencoder(cfg)


    classifier.load_state_dict(torch.load("checkpoints/experiment3/model_1_2_[0.17260980606079102, 0.302325576543808, 0.410335898399353, 0.5010335445404053, 0.5736433863639832].pt"))
    vae.load_state_dict(torch.load("checkpoints/experiment3/model_9_3929_3230.pt"))
    
    model = InferenceModule(cfg, classifier, vae)

    image = Image.open(image_path)
    pose = model.inference(image, target_point)[0].tolist()
    pose = [(round(x[0]), round(x[1])) for x in pose]

    print(pose)

    vis_pose(image_path, pose)

if __name__ == '__main__':
    image_path = 'test.jpg'
    inference_demo(image_path)