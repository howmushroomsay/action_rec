import os
import cv2
import mediapipe as mp
import numpy as np

from dtw import dtw

def skeleton_tran(blazepose):
    # 将33节点的blazepose骨架转换成ntu的25骨架
    visibility = blazepose.shape[1]
    index = [-1, -1, -1, -1, 12, 14, 16, 22, 11, 13, 15, 21,
             24, 26, 28, 32, 23, 25, 27, 31, -1, 20, 22, 19, 21]
    skeleton = np.zeros([25, visibility])
    for i in range(visibility):
        skeleton[0][i] = (blazepose[23][i] + blazepose[24][i]) / 2
        skeleton[1][i] = (blazepose[11][i] + blazepose[12][i] + blazepose[23][i] + blazepose[24][i]) / 4
        skeleton[2][i] = (blazepose[9][i] + blazepose[10][i] + blazepose[11][i] + blazepose[12][i]) / 4
        skeleton[3][i] = (blazepose[1][i] + blazepose[4][i]) / 2
        skeleton[20][i] = (blazepose[11][i] + blazepose[12][i]) / 2
    for i in range(25):
        if index[i] != -1:
            skeleton[i] = blazepose[index[i]]
    a = skeleton[0].copy()
    for i in range(25):
        if visibility > 3:
            skeleton[i][0:3] -= a[0:3]
        else:
            skeleton[i] -= a
    return skeleton


def skeleton_trans_ntu(landmarks):
    # 将25节点的ntu骨架转换成需要计算的15个关键点
    visibility = landmarks.shape[1]
    index = [0, 12, 13, 14, 16, 17, 18, 4, 5, 6, 8, 9, 10, 20, 3]
    skeleton = np.zeros([int(len(index)), visibility])
    for i in range(len(index)):
        skeleton[i] = landmarks[index[i]]
    return skeleton


def skeleton_from_video(video_path):
    # print(os.path.abspath(video_path))
    # assert os.path.exists(video_path), 'File {} not exist!'.format(video_path)

    mp_pose = mp.solutions.pose

    skeleton_list = []
    cap = cv2.VideoCapture(video_path)
    
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=2,
                      smooth_landmarks=True,
                      enable_segmentation=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_idx = -1
        while cap.isOpened():
            success, image = cap.read()
            frame_idx = frame_idx + 1
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image)  # extract landmarks

            skeleton = np.zeros((33, 3))  # mediapipe pose: 33 keypoints [x, y, z, visibility]
            # world landmarks
            if results.pose_world_landmarks:
                for i in range(33):
                    skeleton[i][0] = results.pose_world_landmarks.landmark[i].x
                    skeleton[i][1] = results.pose_world_landmarks.landmark[i].y
                    skeleton[i][2] = results.pose_world_landmarks.landmark[i].z
                    # skeleton[i][3] = results.pose_world_landmarks.landmark[i].visibility

            skeleton = skeleton_tran(skeleton)

            skeleton = skeleton_trans_ntu(skeleton)

            skeleton_list.append(skeleton)
    cap.release()

    return skeleton_list


def keyframe_extract(skeleton_list, visibility=0.8, movements=0.15):
    """

    :param skeleton_list: [frames, key-points, coordinates], coordinates [x, y, z, visibility(optional)]
    :param visibility: threshold
    :param movements: threshold
    :return: key_frames, skeleton_filter_list
    """
    skeleton_list = np.asarray(skeleton_list)
    (frames, kps, visib) = skeleton_list.shape
    # frames = skeleton_list.shape[0]
    # kps = skeleton_list.shape[1]
    # visib = skeleton_list.shape[2]

    key_frames = []
    skeleton_filter_list = []

    for frame in range(frames - 2):

        skeleton_i = skeleton_list[frame]
        skeleton_ii = skeleton_list[frame + 1]
        skeleton_iii = skeleton_list[frame + 2]
        diff = np.ones([kps, 3])

        diff[:, 0] = cosine_distance(skeleton_i[:, 0:3], skeleton_ii[:, 0:3], keep_dim=True)
        diff[:, 1] = cosine_distance(skeleton_ii[:, 0:3], skeleton_iii[:, 0:3], keep_dim=True)

        # diff[:, 0] = np.linalg.norm((skeleton_i[:, 0:3] - skeleton_ii[:, 0:3]), axis=1)
        if visib == 4:
            diff[:, -1] = (skeleton_i[:, 3] + skeleton_ii[:, 3] + skeleton_iii[:, 3]) / 3

        for idx in range(kps):
            if diff[idx, -1] > visibility:
                if np.mean(diff[idx, 0:2]) > movements:
                    key_frames.append(frame + 1)
                    skeleton_filter_list.append(skeleton_ii)
                    break
        if frame == 0:
            key_frames.append(frame + 1)
            skeleton_filter_list.append(skeleton_ii)

    return key_frames, skeleton_filter_list
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if(np.linalg.norm(vec1) < 1e-4) or (np.linalg.norm(vec2)<1e-4):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def cosine_distance(pose1, pose2, keep_dim=False):
    """
    Function that takes in 2 arguments:
        pose1:
            Description: takes in the input pose
            type expected: list of tuples
        pose2:
            Description: takes in the test pose
            type expected: list of tuples
    Returns:
        cosdist:
            Description: the cosine distance between two poses
            type: float

    """
    assert pose1.size == pose2.size
    assert len(pose1.shape) == len(pose2.shape)
    cosdist = 0
    if len(pose1.shape) == 2:
        dot_product = np.diag(pose1.dot(np.transpose(pose2)))
        pose1_norm = np.linalg.norm(pose1, axis=1) + 1e-6
        pose2_norm = np.linalg.norm(pose2, axis=1) + 1e-6
        cossim = dot_product / (pose1_norm * pose2_norm)
        if keep_dim is False:
            cossim = np.average(cossim)
        cosdist = 1 - cossim
    elif len(pose1.shape) == 1:
        cossim = pose1.dot(np.transpose(pose2)) / ((np.linalg.norm(pose1) + 1e-6) * (np.linalg.norm(pose2)) + 1e-6)
        cosdist = 1 - cossim

    return cosdist


def sim2grade(similarity, grades_threshold=None):
    """

    :param similarity: similarity [0, 1]
    :param grades_threshold: list of the grade threshold
    :return: grade ['A', 'B', 'C', 'D']
    """
    if grades_threshold is None:
        grades_threshold = [0.6, 0.75, 0.85]
    assert ((similarity >= 0) & (similarity <= 1.0))
    if similarity < grades_threshold[0]:
        grade = '差'
    elif similarity < grades_threshold[1]:
        grade = '中'
    elif similarity < grades_threshold[2]:
        grade = '良'
    else:
        grade = '优'
    return grade


def action_evaluation(skeleton_T, skeleton_S, keypoints=None, calibration=True, vis_param=0.8, mov_param=0.001,
                      debug=False):
    """

    :param skeleton_T: skeleton of the Teacher
    :param skeleton_S: skeleton of the Student
    :param keypoints:  keyjoints to be focused
    :param vis_param:  visibility param
    :param mov_param:  movement param
    :return: similarity of the two skeleton sequence
    """
    skeleton_T = np.asarray(skeleton_T)
    skeleton_S = np.asarray(skeleton_S)
    assert len(skeleton_T.shape) == len(skeleton_S.shape)
    if debug is True:
        print('Skeleton data of the Teacher is loaded! There are {} frames.'.format(skeleton_T.shape[0]))
        print('Skeleton data of the Student is loaded! There are {} frames.'.format(skeleton_S.shape[0]))

    # step 1
    vec1 = skeleton_S[0, 13, 0:3]  # 13 shoulder center
    vec2 = skeleton_T[0, 13, 0:3]  # 13 shoulder center
    rotation_matrix = rotation_matrix_from_vectors(vec1, vec2)  # vec1 to vec2
    if debug is True:
        print('Rotation Matrix is:')
        print(rotation_matrix)

    # step 2
    if keypoints is not None:
        skeleton_T_kps = skeleton_T[:, keypoints, :]
        skeleton_S_kps = skeleton_S[:, keypoints, :]
    else:
        skeleton_T_kps = skeleton_T
        skeleton_S_kps = skeleton_S

    frames_T, skeleton_T_filted = keyframe_extract(skeleton_T_kps, visibility=vis_param, movements=mov_param)
    frames_S, skeleton_S_filted = keyframe_extract(skeleton_S_kps, visibility=vis_param, movements=mov_param)
    if debug is True:
        print('There are {} key frames from Teacher'.format(np.asarray(skeleton_T_filted).shape[0]))
        print('There are {} key frames from Student'.format(np.asarray(skeleton_S_filted).shape[0]))
    # draw_skeleton_our(np.asarray(skeleton_T[frames_T]))
    # draw_skeleton_our(np.asarray(skeleton_S[frames_S]))
    skeleton_T_filted = np.asarray(skeleton_T_filted)
    skeleton_S_filted = np.asarray(skeleton_S_filted)

    # step 3
    if calibration is True:
        skeleton_S_filted_rotated = np.zeros_like(skeleton_S_filted)
        for frame_index in range(skeleton_S_filted.shape[0]):
            skeleton_S_filted_frame = skeleton_S_filted[frame_index, :, :]
            skeleton_S_filted_frame_rotated = np.zeros_like(skeleton_S_filted_frame)
            for kp_index in range(skeleton_S_filted_frame.shape[0]):
                skeleton_S_frame_kp = skeleton_S_filted_frame[kp_index, 0: 3]
                skeleton_S_frame_kp_rotated = rotation_matrix.dot(skeleton_S_frame_kp)
                skeleton_S_filted_frame_rotated[kp_index, 0: 3] = skeleton_S_frame_kp_rotated
                if skeleton_S_filted_frame.shape[1] > 3:
                    skeleton_S_filted_frame_rotated[kp_index, 3] = skeleton_S_filted_frame[kp_index, 3]
            skeleton_S_filted_rotated[frame_index, :, :] = skeleton_S_filted_frame_rotated

        skeleton_S_filted = skeleton_S_filted_rotated

    # step 4
    warp_param = min(5, int((skeleton_S_filted.shape[0]) / (skeleton_T_filted.shape[0] + 1e-6)))
    warp_param = max(5, warp_param)
    d, cost_matrix, acc_cost_matrix, path = dtw(np.asarray(skeleton_T_filted),
                                                np.asarray(skeleton_S_filted),
                                                dist=cosine_distance,
                                                warp=warp_param,
                                                s=1.0)

    path_valid = 0
    skeleton_T_filted_len = skeleton_T_filted.shape[0]
    cost_list = []
    match_T = []
    match_S = []
    for kf_idx in range(skeleton_T_filted_len):
        index = np.where(path[0] == kf_idx)[0]

        index_len = len(index)
        cost = 0
        for idx in range(index_len):
            p_T = int(path[0][index[idx]])
            p_S = int(path[1][index[idx]])
            cost += cost_matrix[p_T][p_S]
            # print(p_T, p_S, cost_matrix[p_T][p_S])
        cost = cost / index_len

        match_T.append(kf_idx)
        match_S.append(path[1][index[-1]])
        cost_list.append(cost)
        if cost < 0.05:
            path_valid += 1
    similarity = path_valid / skeleton_T_filted_len
    grade_t = [0.6, 0.72, 0.85]
    grade = sim2grade(similarity, grade_t)

    # step 5
    cost_list = np.asarray(cost_list)
    index = list(np.where(cost_list > 0.05)[0])
    disp_T = []
    disp_S = []
    if len(index) > 10:
        fl = len(index) / 10
        for idx in range(10):
            index_f = int(fl*idx)
            disp_T.append(frames_T[match_T[index[index_f]]])
            disp_S.append(frames_S[match_S[index[index_f]]])
    else:
        for idx in range(len(index)):
            disp_T.append(frames_T[match_T[index[idx]]])
            disp_S.append(frames_S[match_S[index[idx]]])

    return similarity, grade, disp_T, disp_S




