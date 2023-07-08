import numpy as np
# from .lib.camera_param import camera_param
camera_param = {
    "CameraParameters1_IntrinsicMatrix" : np.array([[1.219743372126446e+03,0,0],
                                                    [0,1.227032248675082e+03,0],
                                                    [9.901131271084836e+02,5.364585675558315e+02,1]]),

    "CameraParameters1_Distortion":np.array([-0.492956514599428,0.217386309659750,0,0]),

    "CameraParameters2_IntrinsicMatrix" :np.array([[1.228393106018411e+03,0,0],
                                                    [0,1.240361516804724e+03,0],
                                                    [9.255303759171372e+02,5.277266455643949e+02,1]]),

   "CameraParameters2_Distortion":np.array([-0.531418174603698,0.363338006567971,0,0]),

    "R":np.array([[0.680195534177484,-0.358656310052237,0.639296243180516],
                  [0.322717533724995,0.929579059577556,0.178146471818454],
                  [-0.658169756758440,0.085137672358484,0.748040204824524]]),

    "T":np.array([1.989611462530493e+03,-3.558612781835079e+02,9.447364899667228e+02])
}

'''
Camera coordinates to Pix coordinates
Param:
    joints_3D_camera: ndarray n_joints*3
    fx, fy, cx, cy: float
Formula:
    u = fx*x/z+cx
    v = fy*y/z+cy
'''
def _weak_project(joints_3D_camera, fx, fy, cx, cy):
    pose2d = joints_3D_camera[:, :2] /joints_3D_camera[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

def uv2xyz_ourcamera(lx, ly, rx, ry, camera_param):
    # 获取相机参数转换
    # 相机1
    leftIntrinsic = camera_param["CameraParameters1_IntrinsicMatrix"].T
    # 相机2
    rightIntrinsic = camera_param["CameraParameters2_IntrinsicMatrix"].T
    R = camera_param["R"].T
    T = camera_param["T"]

    # 构建约束线性方程
    A = np.zeros(shape=(4, 3))
    A[0] = np.array([leftIntrinsic[0][0],0,leftIntrinsic[0][2]-lx])
    A[1] = np.array([0,leftIntrinsic[1][1],leftIntrinsic[1][2]-ly])
    A[2] = rightIntrinsic[0][0]*R[0,:]-(rx-rightIntrinsic[0][2])*R[2,:]
    A[3] = rightIntrinsic[1][1]*R[1,:]-(ry-rightIntrinsic[1][2])*R[2,:]

    B = np.zeros(shape=(4, 1))
    B[0][0] = 0
    B[1][0] = 0
    B[2][0] = T[2]*(rx-leftIntrinsic[0][2])-leftIntrinsic[0][0]*T[0]
    B[3][0] = T[2]*(ry-leftIntrinsic[1][2])-leftIntrinsic[1][1]*T[1]

    # 采用最小二乘法求其空间坐标
    xyz, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)
    return xyz

if __name__ == "__main__":
    print(uv2xyz_ourcamera(223,456,222,456,camera_param))


