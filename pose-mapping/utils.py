import torch
import os
import cv2
import matplotlib.pyplot as plt


def inference(model, device, x):
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        pred = model(x)
        
    return pred[:10]


def normalize_pose(pose, scale=2):
    
    keypoints_x = [point[0] for point in pose]
    keypoints_y = [point[1] for point in pose]

    min_x = min(keypoints_x)
    max_x = max(keypoints_x)
    min_y = min(keypoints_y)
    max_y = max(keypoints_y)
    
    height = max_y - min_y
    width = max_x - min_x

    if height > width:
        keypoints_x = [((x - min_x) / height - 0.5) * scale for x in keypoints_x]
        keypoints_y = [((y - min_y) / height - 0.5) * scale for y in keypoints_y]
        original_scale = height
    else:
        keypoints_x = [((x - min_x) / width - 0.5) * scale for x in keypoints_x]
        keypoints_y = [((y - min_y) / width - 0.5) * scale for y in keypoints_y]
        original_scale = width

    
    print('keypoints_x = "', keypoints_x, '"')
    print('keypoints_y = "', keypoints_y, '"\n')

    pose = [(x, y) for x, y in zip(keypoints_x, keypoints_y)]

    return pose, min_x, min_y, original_scale, original_scale

def denormalize_pose(pose, min_x, min_y, width, height, scale=2):
    pose = [((x / scale + 0.5) * width + min_x, (y / scale + 0.5) * height + min_y) for (x, y) in pose]

    return pose

def resize_pose(pose, resize_width=400, resize_height=600, padding=10, scale=1.0):
    pose, _, _, _, _ = normalize_pose(pose, scale)
    pose = [((x / scale + 0.9) * resize_width + padding, (y / scale + 0.9) * resize_height + padding) for (x, y) in pose]

    return pose


'''
coco (original):
0 : nose
1 2 : eyes
3 4 : ears
5 7 9 : left arm
6 8 10: right arm
11 13 15: left leg
12 14 16: right leg

mpii (ours):
13 14 15 : left arm
12 11 10 : right arm
3 4 5 : left leg
2 1 0 : right leg
7 : neck
9 8 : head

coco (ours):
0 : nose
1 : neck
2 3 4 : right arm
5 6 7 : left arm
8 9 10 : right leg
11 12 13 : left leg
14 15 : eyes
16 17 : ears
'''

indices = [0, 1, 2, 3, 4, 5, 6, 11, 12]

def access_elements(data, indices):
    return [data[i] for i in indices]

def convert_pose(model, device, pose):
    # re-order keypoints to match neural network input
    reordered_pose = [pose[13], pose[12], pose[14], pose[11], pose[15], pose[10], pose[3], pose[2], pose[4], pose[1], pose[5], pose[0]]

    upper_body = access_elements(reordered_pose, [0, 1, 6, 7]) + [((pose[8][0] + pose[9][0]*2)/3, (pose[8][1] + pose[9][1]*2)/3)]
    upper_body, min_x, min_y, width, height = normalize_pose(upper_body)

    input_x = [point[0] for point in upper_body[:4]]
    input_y = [point[1] for point in upper_body[:4]]

    input_pose = torch.tensor([input_x + input_y])

    # head inference
    head = inference(model, device, input_pose)[0].cpu().tolist()
    head_points = [(x, y) for x, y in zip(head[:5], head[5:])]

    # denormalize head keypoints
    head_points = denormalize_pose(head_points, min_x, min_y, width, height)

    # insert head keypoints to the pose
    reordered_pose = [head_points[0], pose[7], pose[12], pose[11], pose[10], pose[13], pose[14], pose[15], pose[2], pose[1], pose[0], pose[3], pose[4], pose[5],
                      head_points[1], head_points[2], head_points[3], head_points[4]]

    return reordered_pose


mpii_link_pairs = [[0, 1], [1, 2], [2, 6], 
              [3, 6], [3, 4], [4, 5], 
              [6, 7], [7,12], [11, 12], 
              [10, 11], [7, 13], [13, 14],
              [14, 15],[7, 8],[8, 9]]

mpii_link_color = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
              (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (0, 255, 255), (0, 0, 255), (0, 0, 255),
              (0, 0, 255), (0, 255, 0), (0, 255, 0),
              (0, 255, 0), (0, 255, 255), (0, 255, 255)]

mpii_point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(255,255,255)]


coco_link_pairs = [[0, 1], [1, 2], [2, 3], 
              [3, 4], [1, 5], [5, 6], 
              [6, 7], [1, 8], [8, 9], 
              [9, 10], [1, 11], [11, 12],
              [12, 13],[0, 14],[14, 16], [0, 15], [15, 17]]

coco_link_color = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
              (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (0, 255, 255), (0, 0, 255), (0, 0, 255),
              (0, 0, 255), (0, 255, 0), (0, 255, 0),
              (0, 255, 0), (0, 255, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)]

coco_point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(255,255,255), (128, 128, 0), (128, 0, 128)]

'''
mpii (ours):
2 1 0 : right leg
3 4 5 : left leg
12 11 10 : right arm
13 14 15 : left arm
7 : neck
9 8 : head

coco (ours):
0 : nose
1 : neck
2 3 4 : right arm
5 6 7 : left arm
8 9 10 : right leg
11 12 13 : left leg
14 15 : eyes
16 17 : ears
'''

def vis_pose(image_path, pose, link_pairs, link_color, point_color):
    image = cv2.imread(image_path)

    pose = [(int(x), int(y)) for (x, y) in pose]

    for idx, pair in enumerate(link_pairs):
        if pose[pair[0]] != (0, 0) and pose[pair[1]] != (0, 0):
            cv2.line(image, pose[pair[0]], pose[pair[1]], link_color[idx], 2)

    for idx, point in enumerate(pose):
        if point != (0, 0):
            cv2.putText(image, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(image, point, 5, point_color[idx], thickness=-1)

    cv2.imshow("image", image)
    cv2.moveWindow("image", 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    '''


def vis_pose_mpii(image_path, pose):
    vis_pose(image_path, pose, mpii_link_pairs, mpii_link_color, mpii_point_color)


def vis_pose_coco(image_path, pose):
    vis_pose(image_path, pose, coco_link_pairs, coco_link_color, coco_point_color)

