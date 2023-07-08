import numpy as np
import os
import cv2


class Camera_Tools():

    def __init__(self):
        pass

    def _weak_project(self, joints_3D_camera, IntrinsicMatrix):
        '''
        Camera coordinates to Pix coordinates
        Param:
            joints_3D_camera: ndarray n_joints*3
            fx, fy, cx, cy: float
        Formula:
            u = fx*x/z+cx
            v = fy*y/z+cy
        '''
        pose2d = joints_3D_camera[:, :2] / joints_3D_camera[:, 2:3]
        pose2d[:, 0] *= IntrinsicMatrix[0][0]
        pose2d[:, 1] *= IntrinsicMatrix[1][1]
        pose2d[:, 0] += IntrinsicMatrix[2][0]
        pose2d[:, 1] += IntrinsicMatrix[2][1]

        return pose2d

    def leftxyz2rightxyz(self,left_joint_3d,camera_param):
        R = camera_param["R"]
        T = camera_param["T"].T
        return np.matmul(left_joint_3d, R)+T

    def uv2xyz_leftcamera(self, lx, ly, rx, ry, camera_param):
        '''
        返回left相机的相机坐标
        '''
        leftIntrinsic = camera_param["CameraParameters1_IntrinsicMatrix"].T
        rightIntrinsic = camera_param["CameraParameters2_IntrinsicMatrix"].T
        R = camera_param["R"].T
        T = camera_param["T"]

        A = np.zeros(shape=(4, 3))
        A[0] = np.array([leftIntrinsic[0][0], 0, leftIntrinsic[0][2] - lx])
        A[1] = np.array([0, leftIntrinsic[1][1], leftIntrinsic[1][2] - ly])
        A[2] = rightIntrinsic[0][0] * R[0, :] - (rx - rightIntrinsic[0][2]) * R[2, :]
        A[3] = rightIntrinsic[1][1] * R[1, :] - (ry - rightIntrinsic[1][2]) * R[2, :]

        B = np.zeros(shape=(4, 1))
        B[0][0] = 0
        B[1][0] = 0
        B[2][0] = T[2] * (rx - leftIntrinsic[0][2]) - leftIntrinsic[0][0] * T[0]
        B[3][0] = T[2] * (ry - leftIntrinsic[1][2]) - leftIntrinsic[1][1] * T[1]

        xyz, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)

        return xyz

    def getImageList(self, img_dir):
        imgPath = []
        if os.path.exists(img_dir) is False:
            print('error dir')
        else:
            for parent, dirNames, fileNames in os.walk(img_dir):
                for fileName in fileNames:
                    imgPath.append(os.path.join(parent, fileName))
        return imgPath

    def getObjectPoints(self, m, n, k):
        objP = np.zeros(shape=(m * n, 3), dtype=np.float32)
        for i in range(m * n):
            objP[i][0] = i % m
            objP[i][1] = int(i / m)
        return objP * k

    def GetCameraParam(self, path1, path2, x_num, y_num, block_size):
        '''
        :param path1: 图片路径
        :param path2: 图片路径
        :param x_num: 棋盘格行数
        :param y_num: 棋盘格列数
        :param block_size: 棋盘格格子大小 单位：mm
        :return: 双目相机参数字典
        '''
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # 计算标定板真实坐标，标定板内点12*15，大小75mm*75mm
            objPoint = self.getObjectPoints(x_num - 1, y_num - 1, block_size)
            objPoints = []
            imgPointsL = []
            imgPointsR = []
            camera_param = {}
            filePathL = self.getImageList(path1)
            filePathR = self.getImageList(path2)

            for i in range(len(filePathL)):
                # 读取图片并转化为灰度图
                grayL = cv2.cvtColor(cv2.imread(filePathL[i]), cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(cv2.imread(filePathR[i]), cv2.COLOR_BGR2GRAY)
                # opencv寻找角点
                retL, cornersL = cv2.findChessboardCorners(grayL, (x_num - 1, y_num - 1), None)
                retR, cornersR = cv2.findChessboardCorners(grayR, (x_num - 1, y_num - 1), None)
                if (retL & retR) is True:
                    # opencv对真实坐标格式要求，vector<vector<Point3f>>类型
                    objPoints.append(objPoint)
                    # 角点细化
                    cornersL2 = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), criteria)
                    cornersR2 = cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1), criteria)
                    imgPointsL.append(cornersL2)
                    imgPointsR.append(cornersR2)

            # 对左右相机分别进行单目相机标定
            retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL,
                                                                           (1920, 1080), None, None)
            retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR,
                                                                           (1920, 1080), None, None)
            # 双目相机校正
            retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL,
                                                                       imgPointsR, cameraMatrixL,
                                                                       distMatrixL, cameraMatrixR,
                                                                       distMatrixR, (1920, 1080),
                                                                       criteria_stereo,
                                                                       flags=cv2.CALIB_USE_INTRINSIC_GUESS)

            camera_param["CameraParameters1_IntrinsicMatrix"] = np.array(cameraMatrixL).T
            camera_param["CameraParameters2_IntrinsicMatrix"] = np.array(cameraMatrixR).T
            camera_param["R"] = np.array(R).T
            camera_param["T"] = np.array(T)
            return camera_param
        except:
            print("Error: Fail to Get Camera Parameters")

    def GetCameraParam_from_dict(self, img_dict, x_num, y_num, block_size):
        '''
        :param dict: 图片路径
        :param x_num: 棋盘格行数
        :param y_num: 棋盘格列数
        :param block_size: 棋盘格格子大小 单位：mm
        :return: 双目相机参数字典
        '''
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # 计算标定板真实坐标，标定板内点12*15，大小75mm*75mm
            objPoint = self.getObjectPoints(x_num - 1, y_num - 1, block_size)
            objPoints = []
            imgPointsL = []
            imgPointsR = []
            camera_param = {}

            for v in img_dict.values():
                # 读取图片并转化为灰度图
                img1, img2 = v
                grayL = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                # opencv寻找角点
                retL, cornersL = cv2.findChessboardCorners(grayL, (x_num - 1, y_num - 1), None)
                retR, cornersR = cv2.findChessboardCorners(grayR, (x_num - 1, y_num - 1), None)
                if (retL & retR) is True:
                    # opencv对真实坐标格式要求，vector<vector<Point3f>>类型
                    objPoints.append(objPoint)
                    # 角点细化
                    cornersL2 = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), criteria)
                    cornersR2 = cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1), criteria)
                    imgPointsL.append(cornersL2)
                    imgPointsR.append(cornersR2)

            # 对左右相机分别进行单目相机标定
            retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL,
                                                                           (1920, 1080), None, None)
            retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR,
                                                                           (1920, 1080), None, None)
            # 双目相机校正
            retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL,
                                                                       imgPointsR, cameraMatrixL,
                                                                       distMatrixL, cameraMatrixR,
                                                                       distMatrixR, (1920, 1080),
                                                                       criteria_stereo,
                                                                       flags=cv2.CALIB_USE_INTRINSIC_GUESS)

            camera_param["CameraParameters1_IntrinsicMatrix"] = np.array(cameraMatrixL).T
            camera_param["CameraParameters2_IntrinsicMatrix"] = np.array(cameraMatrixR).T
            camera_param["R"] = np.array(R).T
            camera_param["T"] = np.array(T)
            return camera_param
        except:
            return -1

    def get_one_camara_param(self, path1, x_num, y_num, block_size):
        '''
        :param path1: 图片路径
        :param x_num: 棋盘格行数
        :param y_num: 棋盘格列数
        :param block_size: 棋盘格格子大小 单位：mm
        :return: 单目相机参数字典
        '''
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objPoint = self.getObjectPoints(x_num - 1, y_num - 1, block_size)
            objPoints = []
            imgPoints = []
            camera_param = {}
            filePath = self.getImageList(path1)

            for i in range(len(filePath)):
                grayL = cv2.cvtColor(cv2.imread(filePath[i]), cv2.COLOR_BGR2GRAY)
                retL, cornersL = cv2.findChessboardCorners(grayL, (x_num - 1, y_num - 1), None)
                if retL is True:
                    objPoints.append(objPoint)
                    cornersL2 = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), criteria)
                    imgPoints.append(cornersL2)

            retL, cameraMatrix, distMatrix, R, T = cv2.calibrateCamera(objPoints, imgPoints,
                                                                       (1920, 1080), None, None)

            camera_param["CameraMatrix"] = np.array(cameraMatrix).T
            camera_param["DistMatrixL"] = np.array(distMatrix)
            camera_param["R"] = np.array(R).T
            camera_param["T"] = np.array(T)

            return camera_param
        except:
            print("Error: Fail to Get Camera Parameters")
