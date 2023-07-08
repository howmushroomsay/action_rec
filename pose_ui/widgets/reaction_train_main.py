import json
import os
import sys
import time
import requests
import datetime
from multiprocessing import Event, Process, Queue

from PyQt5.QtWidgets import QApplication,QWidget, QGraphicsOpacityEffect
from PyQt5 import  QtCore, Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from .ui import Ui_Reaction_Train_PlayerP1

from auxiliary_tools import skeleton_get, seg, actionRec, parse_data
from .reaction_train_contral import Reaction_Train_Player_Contral, Reaction_Result, Reaction_Train_Player_Frame


class Reaction_Train_Main(QWidget, Ui_Reaction_Train_PlayerP1):
    def __init__(self, parent, db, course_id, student_id):
        super(Reaction_Train_Main, self).__init__()
        self.parent = parent
        self.db = db
        self.course_id = course_id
        self.student_id = student_id      
        self.setupUi(self)
        self.windowinit()

        # self.contral = Reaction_Train_Player_Contral(self)
        self.hint_frame = Reaction_Train_Player_Frame(200,self)
        # self.contral_ifshow = False
        # self.contral.show()
        
        self.load_config()
        self.start_process()
        self.getCourseInfo()
        self.initfun()      

        
    def load_config(self, json_path='./data/sensor/config.json'):
        with open(json_path, encoding='UTF-8') as f:
            config = json.load(f)
        if config["using_sensor"] == 'F':
            self.using_sensor = False
        else:
            self.using_sensor = True
        if len(config["camera"]) == 1:
            self.camera = int(config["camera"])
        else:
            self.camera = config["camera"]

    def windowinit(self):
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setMouseTracking(True)
        # self.player_frame.setMouseTracking(True)
        self.videoWidget.setMouseTracking(True)

    def initfun(self):
        self.play_flag = True
        self.num = 5
        
        self.tip_timer = QtCore.QTimer()
        self.tip_timer.timeout.connect(self.change_num)
        self.timer_stamp = QtCore.QTimer()
        self.timer_stamp.timeout.connect(self.timecheck)
        self.timer_show = QtCore.QTimer()
        self.timer_show.timeout.connect(self.show_result)
        self.point_start_flag = True
        self.point_in_flag = False
        self.action_l = []
        self.tip_timer.start(1000)

    def get_ip(self):
        json_path = "./data/dbconfig/database_account_password.json"
        with open(json_path) as f:
            data_base_json = json.load(f)        
        return data_base_json["video_ip"], data_base_json["course_dir"]
    
    def getCourseInfo(self):

        # self.course_path = '../../test/格斗视频/格斗视频3.mp4'
        # self.points = [ [50, 1496, 2897, 6], [51, 7519, 8867, 3], [52, 12296, 13150, 2],
        #                 [53, 21883, 25070, 6], [54, 30392, 32662, 3], [55, 38505, 40751, 2],
        #                 [56, 50710, 52788, 5]]

        #获取课程视频路径
        sql_path = """SELECT course_Path
                    FROM course_actionresponse
                    WHERE course_ID = {}""".format(self.course_id)
        self.course_path = self.db.search_table(sql_path)[0][0]
        
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
        # 获取检查点信息
        sql_point = """SELECT point_ID, start_time, end_time, action_ID
                        FROM point_actionresponse
                        WHERE course_ID = {}""".format(self.course_id)
        self.points = self.db.search_table(sql_point)
        # 勾拳 踢腿 勾拳 踢腿 闪躲 闪躲 提膝
        self.points_index = 0
        self.action_index = 0

    def ready(self):
        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.videoWidget)
        media = QtCore.QUrl.fromLocalFile(self.course_path)
        self.player.setMedia(QMediaContent(media))
        self.player.play()
        self.timer_stamp.start(25)
        self.timer_show.start(25)
    def change_num(self):
        opacity = QGraphicsOpacityEffect()
        self.lab_tip.setStyleSheet("color: rgb(255, 255, 255);\n"
                                     "background-color: rgb(0, 0, 0);\n"
                                     "font: 100pt \"Pristina\";")
        if self.num > 0:
            for i in range(self.num*10, self.num*10 - 10, -1):
                self.lab_tip.setText(str(self.num))
                self.lab_tip.repaint()
                opacity.setOpacity(i / 50)
                self.lab_tip.setGraphicsEffect(opacity)
                time.sleep(0.05)
            self.num -= 1
        elif self.num == 0:
            self.lab_tip.setText("开始！")
            self.lab_tip.repaint()
            self.num -=1
        else:
            self.lab_tip.hide()
            self.tip_timer.stop()
            self.ready()

    def show_result(self):
        # 直到取出最后一份结果再展示
        if self.acqueue.qsize():
            start, end, action, pro = self.acqueue.get()
            if pro > 0.5:
                self.action_l.append((start, end, action, pro))
            else:
                self.action_l.append((0, 0, -1, 1))
            if pro > 0.5 and action == self.points[self.action_index][3]:
                self.hint_frame.show_green() 
            else:
                self.hint_frame.show_red()
            self.action_index += 1
        if (len(self.action_l) == len(self.points)):
            self.timer_show.stop()
        else:
            return
        target = [self.points[i][3] for i in range(len(self.points))]
        self.result_ = Reaction_Result(self, target, self.action_l)
        self.result_.show()

    def timecheck(self):
        if self.points_index == len(self.points) or not self.play_flag:
            self.timer_stamp.stop()
            return
    
        current_time = self.player.position()

        if current_time + 200 >= self.points[self.points_index][1] and self.point_start_flag:
            #提示检查点开始
            self.hint_frame.show_blue()
            self.point_start_flag = False

        if self.points[self.points_index][2] > current_time > self.points[self.points_index][1]:
            if self.point_in_flag:
                self.timequeue.put((current_time, 0))
            else:
                self.timequeue.put((current_time, -2))
                self.point_in_flag = True
        if current_time >= self.points[self.points_index][2] and self.point_in_flag:
            self.timequeue.put((current_time, -1))
            self.point_in_flag = False
            self.point_start_flag = True
            self.points_index += 1

    # def changeSlide(self, position):
    #     self.videoLength = self.player.duration() + 0.1
    #     self.contral.sld_video.setValue(round((position/self.videoLength)*100))
    #     m, s = divmod(self.player.position() / 1000, 60)
    #     h, m = divmod(m, 60)
    #     a = "%02d:%02d:%02d" % (h, m, s)
    #     m, s = divmod(self.player.duration() / 1000, 60)
    #     h, m = divmod(m, 60)
    #     a += "/%02d:%02d:%02d" % (h, m, s)
    #     self.contral.lab_time.setText(a)
    
    # def changestate(self):
    #     if self.play_flag:
    #         self.player.pause()
    #     else:
    #         self.player.play()
    #     self.play_flag =  not self.play_flag

    def start_process(self):
        self.stop_event = Event()
        self.timequeue = Queue(10)
        self.corqueue = Queue(10)
        self.skqueue = Queue(10)
        self.acqueue = Queue(10)

        process_id = []
        if self.using_sensor:
            process_id.append(Process(target=parse_data, 
                                      args=(self.stop_event, 
                                            self.timequeue, 
                                            self.corqueue)))
        else:
            process_id.append(Process(target=skeleton_get,
                                      args=(self.stop_event, 
                                            self.timequeue,
                                            self.corqueue,
                                            self.camera)))
        process_id.append(Process(target=seg,
                                  args=(self.stop_event,
                                        self.corqueue, 
                                        self.skqueue))) 
        process_id.append(Process(target=actionRec, 
                                  args=(self.stop_event,
                                        self.skqueue,
                                        self.acqueue)))

        for i in process_id:
            i.daemon = True
            i.start()
            
    def exit(self, score):
        if score != 0:
            sql_upload = """INSERT grade_reaction
                            (student_ID, course_ID, grade, train_time)
                            VALUES(%s, %s, %s, %s)"""
            t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            args = [[self.student_id, self.course_id,  score, t]]
            self.db.insert_data(sql=sql_upload, args=args)                    
        self.player.stop()            
        self.stop_event.set()

        self.hint_frame.timer_end.stop()
        # self.parent.show()
        self.hint_frame.close()
        # self.contral.close()
        self.close()
    
    # def mouseMoveEvent(self, evt):
    #     if evt.pos().y() > 900:
    #         if not self.contral_ifshow:
    #             self.contral.draw_show()
    #             self.contral_ifshow = True
    #     else:
    #         if self.contral_ifshow:
    #             self.contral.draw_hide()
    #             self.contral_ifshow = False


if __name__ == "__main__":

    app = QApplication(sys.argv)
    mainWindow = Reaction_Train_Main()
    mainWindow.show()
    sys.exit(app.exec_())




