import os
import sys
import json
import requests
import numpy as np

from PyQt5.QtWidgets import QApplication,QWidget,QGridLayout
from PyQt5 import QtCore
from PyQt5.Qt import *
from PyQt5.QtGui import QMovie,QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt




from .ui import Ui_Action_FollowP1,Ui_Action_FollowP2,Ui_Action_FollowP3


class Figure_Canvas(FigureCanvas):
    def __init__(self,width=400,height=400,dpi=100):
        self.fig=Figure(figsize=(width,height),dpi=dpi)#设置长宽以及分辨率
        super(Figure_Canvas,self).__init__(self.fig)
        self.ax = self.fig.add_subplot(projection='3d')#创建axes对象实例，这个也可以在具体函数中添加
        self.ax.axis('off')
        self.ax.grid(None)
        # self.ax.set_title()
        
        
class Action_Follow_TOP(QWidget,Ui_Action_FollowP2):

    def __init__(self, parent):
        super(Action_Follow_TOP, self).__init__()
        self.setupUi(self)
        self.windowinit()
        self.parent = parent
        
        # 初始不隐藏，暂停状态
        self.hide_flag = False
        self.all_hide_flag = False
        self.video_play = False

        self.initplot()
        self.btn_exit.clicked.connect(self.all_close)
        # self.btn_hide_control.clicked.connect(self.hide_control)
        self.btn_replay.clicked.connect(self.replay)
        self.btn_stoporplay.clicked.connect(self.change_state)
        # self.btn_bar_hide.clicked.connect(self.parent.Plot)
        # self.hide_control()
    def replay(self):
        if self.parent.player.position() != self.parent.player.duration():
            return
        self.parent.Play()
        self.parent.Play()
        self.parent.keyframe_index = 0
        self.lab_text.setText('')
    def windowinit(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.SubWindow | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
    
    def all_close(self):
        self.close()
        self.parent.back()
        # self.parent.video_bar.close()

    def change_state(self):
        if self.parent.player.position() == self.parent.player.duration():
            return
        if self.parent.play_flag:
            self.parent.Pause()
        else:
            if self.parent.stop_count > 0:
                return
            else:
                self.parent.Play() 
        self.play_control()

    def play_control(self):
        # 改变图标用
        if self.parent.play_flag:
            pix_img = QPixmap(":/icons/figs/icon/menu_stop.png")
            pix_img = pix_img.scaled(50, 50, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.btn_stoporplay.setIcon(QIcon(pix_img))
        else:
            pix_img = QPixmap(":/icons/figs/icon/menu_play.png")
            pix_img = pix_img.scaled(35, 35, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.btn_stoporplay.setIcon(QIcon(pix_img))
            
    # def hide_control(self):
    #     if self.hide_flag:
    #         # self.control_frame.move(1590, 900)
    #         self.control_frame.move(1720, 900)
    #     else:
    #         # self.control_frame.move(1870, 900)
    #         self.control_frame.move(1870, 900)
    #     self.hide_flag = not self.hide_flag

    def all_hide(self):
        if self.all_hide_flag:
            # self.lab_3dpose.show()
            self.lab_text.show()
            self.lab_title.show()
        else:
            # self.lab_3dpose.hide()
            self.lab_text.hide()
            self.lab_title.hide()
        self.all_hide_flag = not self.all_hide_flag

    def initplot(self):
        self.plot_Figure = Figure_Canvas()
        self.plot_FigureLayout = QGridLayout(self.pose_frame)#利用栅格布局将图像与画板连接
        self.plot_FigureLayout.addWidget(self.plot_Figure)
        
    def plot(self, skeleton):
        self.plot_Figure.ax.cla()
        self.plot_Figure.ax.axis('off')
        self.plot_Figure.ax.grid(None)
        self.plot_Figure.ax.set_xlim3d([-1, 1])
        self.plot_Figure.ax.set_ylim3d([-1, 1])
        self.plot_Figure.ax.set_zlim3d([-0.2, 1.8])
        # self.plot_Figure.ax.set_xlabel('X')
        # self.plot_Figure.ax.set_ylabel('Z')
        # self.plot_Figure.ax.set_zlabel('Y')
        self.plot_Figure.ax.view_init(0, -90)

        x = 2*skeleton[:, 0]
        y = -2*skeleton[:, 1] + 0.5
        z = 2*skeleton[:, 2]
        body = [[0,13,14], [6,5,4,0,1,2,3],[12,11,10,13,7,8,9]]
        for part in body:
            x_plot = x[part]
            y_plot = y[part]
            z_plot = z[part]
            self.plot_Figure.ax.plot(x_plot, z_plot, y_plot, color='b',
                    marker='o', markerfacecolor='r')
        self.plot_Figure.draw()

class Action_Follow_Main(QWidget,Ui_Action_FollowP1):

    def __init__(self, parent, db, action_id):
        super(Action_Follow_Main, self).__init__()
        self.setupUi(self)
        self.windowinit()

        self.control_widget = Action_Follow_TOP(self)
        # self.video_bar = Action_Follow_Bar()
        # self.video_bar.show()
        self.control_widget.show()

        self.parent = parent
        self.db = db
        self.action_id = action_id
        self.getActionInfo()
        self.initfun()
        
        self.ready()
        self.stop_count = 0
        self.show_flag = False
    
    def windowinit(self):
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

    def initfun(self):
        self.control_widget.lab_title.setText(self.course_Name)
        self.play_flag = True
        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.videoWidget)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timecheck)
    
    def get_ip(self):
        json_path = "./data/dbconfig/database_account_password.json"
        with open(json_path) as f:
            data_base_json = json.load(f)        
        return data_base_json["video_ip"], data_base_json["course_dir"]

    def getActionInfo(self):
        # 获取动作视频路径
        sql_course = """SELECT action_Path, action_Name
                    FROM action_info
                    WHERE action_ID = {}""".format(self.action_id)
        course_info = self.db.search_table(sql_course)
        self.course_path = course_info[0][0]
        self.course_Name = course_info[0][1]      
        
        pos = self.course_path.find('course_action')
        self.video_ip, self.course_dir = self.get_ip()
        course_url = self.course_dir + self.course_path[pos:]
        course_url = "http://" + self.video_ip + course_url
        response = requests.get(course_url)
        self.course_path = './data' + self.course_path[pos-1:]
        
        if not os.path.exists(os.path.dirname(self.course_path)):
            os.makedirs(os.path.dirname(self.course_path))
        if not os.path.exists(self.course_path):
            with open(self.course_path, 'wb') as f:
                f.write(response.content)

        # 获取关键帧信息
        sql_keyframe = """SELECT keyframe_time, keyframe_Description, keyframe_Focus, keyframe_Skeleton
                            FROM keyframe_info
                            WHERE action_id = {}""".format(self.action_id)
        
        keyframe_info = self.db.search_table(sql_keyframe)
        self.keyframe_time = [i[0] for i in keyframe_info]
        self.keyframe_Description = [i[1] for i in keyframe_info]
        self.keyframe_Focus = [i[2] for i in keyframe_info]

        self.keyframe_Skeleton = [np.reshape(np.frombuffer(i[3], dtype=np.float64), (15,3)) for i in keyframe_info]
        
        self.keyframe_index = 0
        # self.action_path = r'E:\PoseUI\PoseUI\data\camera_video_demo\camera_192.168.1.2.mp4'
    
    def ready(self):
        media = QtCore.QUrl.fromLocalFile(self.course_path)
        self.player.setMedia(QMediaContent(media))
        self.Play()
        self.timer.start(25)

    def timecheck(self):
        
        # 每个关键帧展示3s
        if(self.player.position() == self.player.duration()) and self.keyframe_index == len(self.keyframe_time):
            # self.control_widget.plot_Figure.ax.cla()
            # self.control_widget.plot_Figure.ax.axis('off')
            # self.control_widget.plot_Figure.ax.grid(None)
            # self.control_widget.plot_Figure.draw()
            self.control_widget.lab_text.setText('视频播放结束')
        if(self.show_flag):
            if(self.stop_count < 120):
                self.stop_count += 1
                return
            else:
                self.stop_count = 0
                self.Play()
                self.control_widget.play_control()
                self.show_flag = False
        
        # 检查是否到达关键帧时间
        current_time = self.player.position()
        # 播放完毕或者未到关键帧时间返回
        if self.keyframe_index == len(self.keyframe_time):
            return
        if(current_time==self.player.duration() or current_time < self.keyframe_time[self.keyframe_index]):
            return
        self.show_flag = True
        self.stop_count = 1
        self.control_widget.lab_text.setText('动作要领:' + self.keyframe_Description[self.keyframe_index])
        self.Plot()
        self.keyframe_index += 1
        # self.keyframe_index %= len(self.keyframe_time)
        self.Pause()
        self.control_widget.play_control()

    def break_keyframeshow(self):
        # 展示关键帧时通过播放按钮打断
        self.stop_count = 0
        self.Play()

    def Play(self):
        self.player.play()
        self.play_flag = True

    def Pause(self):
        self.player.pause()
        self.play_flag = False

    def Plot(self):
        self.control_widget.plot(self.keyframe_Skeleton[self.keyframe_index])

    def back(self):
        # self.parent.show()
        self.close()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Action_Follow_Main()
    win.show()
    sys.exit(app.exec_())