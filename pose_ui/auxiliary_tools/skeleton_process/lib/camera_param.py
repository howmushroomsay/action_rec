import numpy as np

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

if __name__ == "__main__":
    print(camera_param)
