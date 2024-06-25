import os

import cv2

link_pairs = [[0, 1], [1, 2], [2, 6], 
              [3, 6], [3, 4], [4, 5], 
              [6, 7], [7,12], [11, 12], 
              [10, 11], [7, 13], [13, 14],
              [14, 15],[7, 8],[8, 9]]

link_color = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
              (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (0, 255, 255), (0, 0, 255), (0, 0, 255),
              (0, 0, 255), (0, 255, 0), (0, 255, 0),
              (0, 255, 0), (0, 255, 255), (0, 255, 255)]

point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(255,255,255)]

def vis_pose(image_path, pose_keypoints, show=True):
    image = cv2.imread(image_path)

    for idx, pair in enumerate(link_pairs):
        cv2.line(image, pose_keypoints[pair[0]], pose_keypoints[pair[1]], link_color[idx], 2)

    for idx, point in enumerate(pose_keypoints):
        if idx != 16:
            cv2.circle(image, point, 5, point_color[idx], thickness=-1)
        else:
            cv2.circle(image, point, 20, point_color[idx], thickness=-1)

    if show:
        cv2.imshow("image", image)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def vis_pose_data(data):
    image_dir_path = './affordance_data/data'
    data = data.split(' ')
    image_path = os.path.join(image_dir_path, data[0])
    image_cluster = eval(data[-1])
    pose_data = data[1:-1]
    pose_data = [int(eval(x)) for x in pose_data]
    pose_keypoints = []
    for i in range(0, len(pose_data), 2):
        pose_keypoints.append((pose_data[i], pose_data[i+1]))
    vis_pose(image_path, pose_keypoints)

if __name__ == '__main__':
    predicted = [(595.5921267553274, 404.10781700000007), (587.4540269712579, 375.26168899999993), (674.1323874006324, 382.447135), (709.0841106363025, 390.41310699999985), (738.5057531861758, 386.341047), (664.3825897951702, 417.24250400000005), (693.034285958547, 387.2644009999999), (714.1887366306264, 310.66663199999994), (711.364003176137, 296.50660700000003), (702.7690875859935, 228.0), (630.6202741684986, 387.9697880000001), (647.1081440971391, 359.6237799999999), (670.0994861445556, 309.9926839999998), (761.1125559742334, 314.1373520000001), (758.0154726576313, 389.9528070000001), (678.8184304028331, 402.0585960000001)]
    gt = [(613, 410), (603, 374), (646, 367), (671, 370), (673, 389), (630, 415), (660, 370), (672, 298), (670, 276), (665, 228), (638, 382), (631, 340), (642, 298), (703, 298), (693, 354), (640, 379)]
    predicted = [(round(x[0]), round(x[1])) for x in predicted]

    vis_pose(os.path.join('./affordance_data/data', 'ELR_matches/S02/E0012.mkv/frame_00002381.jpg'), predicted + [(0, 0)])
    vis_pose(os.path.join('./affordance_data/data', 'ELR_matches/S02/E0012.mkv/frame_00002381.jpg'), gt + [(0, 0)])
    input()

    train_data_path = './affordance_data/trainlist.txt'
    train_data = []
    with open(train_data_path, 'r') as f:
        train_data = list(f.readlines())

    for data in train_data:
        vis_pose_data(data)