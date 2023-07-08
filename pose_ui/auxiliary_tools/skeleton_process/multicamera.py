import os
import mediapipe as mp
import multiprocessing
import numpy as np
import cv2
from .lib.camera import uv2xyz_ourcamera, _weak_project
from .lib.camera_param import camera_param
import time
'''
Keypoints:
0-L.shoulder 1-L.elbow 2-L.wrist
3-R.shoulder 4-R.elbow 5-R.wrist
6-L.hip 7-L.knee 8-L.ankle
9-R.hip  10-R.knee 11-R.ankle

Infer keypoints:
12-hip 13-throx 14-head
'''

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def skeleton_tran(c):
    n = [-1, -1, -1, -1, 12, 14, 16, 22, 11, 13, 15, 21,
         24, 26, 28, 32, 23, 25, 27, 31, -1, 20, 22, 19, 21]
    skeleton = np.zeros([25, 3])
    for i in range(3):
        skeleton[0][i] = (c[23][i] + c[24][i]) / 2
        skeleton[1][i] = (c[11][i] + c[12][i] + c[23][i] + c[24][i]) / 4
        skeleton[2][i] = (c[9][i] + c[10][i] + c[11][i] + c[12][i]) / 4
        skeleton[3][i] = (c[1][i] + c[4][i]) / 2
        skeleton[20][i] = (c[11][i] + c[12][i]) / 2
    for i in range(25):
        if n[i] != -1:
            skeleton[i] = c[n[i]]
    a = skeleton[0].copy()
    for i in range(25):
        skeleton[i] -= a
    return skeleton


def get_pose2d(path, q, w=1920, h=1080):
    cap = cv2.VideoCapture(path)
    count = 0
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_RGB)
            keypoints_2d = np.zeros(shape=(33, 2))

            if results.pose_landmarks:
                for i in range(33):
                    keypoints_2d[i][0] = results.pose_landmarks.landmark[i].x * w
                    keypoints_2d[i][1] = results.pose_landmarks.landmark[i].y * h
                q.put((1, keypoints_2d))
            else:
                q.put((0, 0))
                continue
        else:
            break


def get_pose3d(queues, pose3dqueue, camera_param):
    start = time.time()
    end = time.time()
    count = 0
    while end - start < 10:
        if queues[0].qsize() > 1 and queues[1].qsize() > 1:
            start = time.time()
            end = time.time()
            pose2d_1 = queues[0].get()
            pose2d_2 = queues[1].get()
            if pose2d_1[0] != 0 and pose2d_2[0] != 0:
                keypoints_3d = np.zeros((33, 3))
                for j in range(33):
                    keypoints_3d[j] = uv2xyz_ourcamera(lx=pose2d_1[1][j][0], ly=pose2d_1[1][j][1],
                                                       rx=pose2d_2[1][j][0], ry=pose2d_2[1][j][1],
                                                       camera_param=camera_param).squeeze()
                pose3dqueue.put((skeleton_tran(keypoints_3d), 1, True))

                count += 1
            else:
                end = time.time()
                pose3dqueue.put((np.zeros([25, 3]), 1, False))
                # pose3dqueue.put("No Person")
        else:
            end = time.time()
            continue
    print('get timeout')


def get_3d_main(pose3d_queue, cameras=None):
    if cameras:
        camera_info = cameras
    else:
        camera_info = ['./skeleton_process/data/192.168.1.2action_rec.mp4',
                       './skeleton_process/data/192.168.1.64action_rec.mp4']
    for camera in camera_info:
        if os.path.exists(camera):
            continue
        else:
            print(os.path.realpath(camera))
    queues = [multiprocessing.Queue(maxsize=4) for _ in camera_info]
    # pose3d_queue = multiprocessing.Queue(maxsize=4)
    processes = []
    for queue, camera_id in zip(queues, camera_info):
        processes.append(multiprocessing.Process(
            target=get_pose2d, args=(camera_id, queue)))

    processes.append(multiprocessing.Process(
        target=get_pose3d, args=(queues, pose3d_queue, camera_param)))

    for process in processes:
        process.daemon = True
        process.start()
    # for process in processes:
    #     process.join()


if __name__ == '__main__':
    get_3d_main()
