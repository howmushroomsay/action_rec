import numpy as np
import cv2, torch,multiprocessing
import mediapipe as mp
def skeleton_tran(c):
        n = [-1,-1,-1,-1,12,14,16,22,11,13,15,21,24
            ,26,28,32,23,25,27,31,-1,20,22,19,21]
        skeleton = np.zeros([25,3])
        for i in range(3):
            skeleton[0][i]  = (c[23][i] + c[24][i]) / 2
            skeleton[1][i]  = (c[11][i] + c[12][i]  + c[23][i] + c[24][i]) / 4
            skeleton[2][i]  = (c[9][i]  + c[10][i]  + c[11][i] + c[12][i]) / 4
            skeleton[3][i]  = (c[1][i]  + c[4][i])  / 2
            skeleton[20][i] = (c[11][i] + c[12][i]) / 2
        for i in range(25):
            if n[i] != -1:
                skeleton[i] = c[n[i]]
        a = skeleton[0].copy()
        for i in range(25):
            skeleton[i] -= a
        return skeleton
def skeleton_process(skeleton):
    #将连续骨骼序列处理成网络需要的输入
    conn = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
    data = skeleton.transpose(3, 1, 2, 0)
    C, T, V, M = data.shape
    joint = np.zeros((C*2, T, V, M))
    velocity = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    joint[:C,:,:,:] = data
    for i in range(V):
        joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
    for i in range(T-2):
        velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
    for i in range(len(conn)):
        bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += bone[i,:,:,:] ** 2
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
    data_new =[joint,velocity,bone]
    data_new = np.stack(data_new, axis=0)
    return torch.tensor(data_new)
def single_skeleton(skeleton_queue:multiprocessing.Queue,path:str):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    if path:
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(0)
    skeleton = np.zeros([33,3])
    while True:
        success,img = cap.read()
        cv2.imshow('hhh', img)
        cv2.waitKey(1)
        if not success:
            break
        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = pose.process(img_RGB)

        if results.pose_world_landmarks and results.pose_landmarks:
            # 读取blazepose中的骨骼数据保存在列表中
            for i in range(33):
                skeleton[i][0] = results.pose_world_landmarks.landmark[i].x
                skeleton[i][1] = results.pose_world_landmarks.landmark[i].y
                skeleton[i][2] = results.pose_world_landmarks.landmark[i].z
            skeleton_queue.put(skeleton_tran(skeleton)* 1000)


    
    