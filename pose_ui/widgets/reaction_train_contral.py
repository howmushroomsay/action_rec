from PyQt5.QtWidgets import QWidget, QHeaderView, QGraphicsOpacityEffect
from PyQt5 import  QtCore, QtWidgets
from PyQt5.QtGui import QBrush, QFont
from PyQt5.QtCore import Qt
from .ui import Ui_Reaction_Train_PlayerP2,Ui_Reaction_Train_PlayerP3, Ui_Reaction_Result

class Reaction_Train_Player_Contral(QWidget, Ui_Reaction_Train_PlayerP2):

        def __init__(self, parent):
            super(Reaction_Train_Player_Contral, self).__init__()
            self.setupUi(self)
            self.windowinit()
            self.parent = parent
            self.btn_exit.clicked.connect(self.quit)

            self.btn_playorstop.clicked.connect(self.changestate)

            self.opacity = QGraphicsOpacityEffect()
            self.opacity.setOpacity(0)
            self.contrl_frame.setGraphicsEffect(self.opacity)  

        def windowinit(self):
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.SubWindow | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setMouseTracking(True)
            self.contrl_frame.setMouseTracking(True)

        def changestate(self):
            self.parent.changestate()

        def quit(self):
            # return 
            self.parent.exit(0)

        def draw_show(self):
            self.opacity.i = 0
            def timeout():
                self.opacity.setOpacity(self.opacity.i / 100)
                self.contrl_frame.setGraphicsEffect(self.opacity)
                self.opacity.i += 2
                if self.opacity.i >= 100:
                    self.timer.stop()
                    self.timer.deleteLater()

            self.timer = QtCore.QTimer()
            self.timer.setInterval(10)
            self.timer.timeout.connect(timeout)
            self.timer.start()

        def draw_hide(self):
            self.opacity.i = 100
            def timeout():
                self.opacity.setOpacity(self.opacity.i / 100)
                self.contrl_frame.setGraphicsEffect(self.opacity)
                self.opacity.i -= 2
                if self.opacity.i < 0:
                    self.timer.stop()
                    self.timer.deleteLater()

            self.timer = QtCore.QTimer()
            self.timer.setInterval(10)
            self.timer.timeout.connect(timeout)
            self.timer.start()

        def mouseMoveEvent(self, evt):
            if evt.pos().y() > 900:
                if not self.parent.contral_ifshow:
                    self.draw_show()
                    self.parent.contral_ifshow = True
            else:
                if self.parent.contral_ifshow:
                    self.draw_hide()
                    self.parent.contral_ifshow = False

class Reaction_Result(QWidget, Ui_Reaction_Result):
    def __init__(self, parent, target, rec_result) -> None:
        super(Reaction_Result, self).__init__()
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.SubWindow | Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.cal_score(target, rec_result)
        self.show_result(target, rec_result)
        
        self.pushButton.clicked.connect(self.exit)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setVisible(False)
    def cal_score(self, target, rec_result):
        score_matrix_txt = './data/course_action/score_matrix.txt'
        score_matrix1 = [[] for _ in range(9)]
        score_matrix2 = [[] for _ in range(9)]
        with open(score_matrix_txt, 'r') as f:
            for i in range(9):
                temp = f.readline().strip().split()
                for j in range(9):
                    score_matrix1[i].append(int(temp[j]))
            f.readline()
            for i in range(9):
                temp = f.readline().strip().split()
                for j in range(9):
                    score_matrix2[i].append(int(temp[i]))
        score = 0
        for i in range(len(target)):
            score += score_matrix1[target[i]+1][rec_result[i][2]+1]*0.6 
            score += score_matrix2[target[i]+1][rec_result[i][2]+1]*0.4
        score /= len(target)
        if score < 70:
            self.score = '中'
        elif score < 80:
            self.score = '良'
        else:
            self.score = '优'
    def show_result(self, target, rec_result):
        action_class = {-1: '无动作',0: '勾拳', 1: '格挡', 2: '肘击', 3: '直拳',
                    4: '踢腿', 5: '提膝', 6: '戒备', 7: '闪躲'}
        self.tableWidget.horizontalHeader().setStyleSheet(
            "QHeaderView::section{background-color: rgba(23, 56, 114, 200);}")

        self.label.setText('最后得分: {}'.format(self.score))
        self.tableWidget.setRowCount(len(target))
        for i in range(len(target)):
            target_item = QtWidgets.QTableWidgetItem(action_class[target[i]])
            target_item.setForeground(QBrush(Qt.white))
            target_item.setFont(QFont('微软雅黑',16))
            target_item.setTextAlignment(Qt.AlignCenter)
            start, end, action, pro = rec_result[i]
            result_item = QtWidgets.QTableWidgetItem(action_class[action])
            result_item.setForeground(QBrush(Qt.white)) 
            result_item.setTextAlignment(Qt.AlignCenter)
            result_item.setFont(QFont('微软雅黑',16))
            self.tableWidget.setItem(i, 0, target_item)
            self.tableWidget.setItem(i, 1, result_item)
    def exit(self):
        self.close()
        self.parent.exit(self.score)

class Reaction_Train_Player_Frame(QWidget, Ui_Reaction_Train_PlayerP3):
    #提示界面
    def __init__(self, time, parent):
        super(Reaction_Train_Player_Frame, self).__init__()
        self.setupUi(self)
        self.windowinit()

        self.parent = parent
        self.timer_end = QtCore.QTimer(self)
        self.timer_end.timeout.connect(self.quit)
        self.time = time

    def windowinit(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.feed_frame.setMouseTracking(True)

    def quit(self):
        self.timer_end.stop()
        self.hide()

    def show_red(self):
        self.feed_frame.setStyleSheet("QFrame{\n"
        "background-color: qradialgradient(spread:pad, \n"
        "cx:0.5, cy:0.5, radius:0.9, fx:0.5, fy:0.5, \n"
        "stop:0 rgba(0, 0, 0, 0), stop:1 rgba(200, 0, 0, 255)); \n"
        "}")
        self.show()
        self.timer_end.start(self.time)
    
    def show_blue(self):
        self.feed_frame.setStyleSheet("QFrame{\n"
        "background-color: qradialgradient(spread:pad, \n"
        "cx:0.5, cy:0.5, radius:0.9, fx:0.5, fy:0.5, \n"
        "stop:0 rgba(0, 0, 0, 0), stop:1 rgba(0, 0, 200, 255)); \n"
        "}")
        self.show()
        self.timer_end.start(self.time)

    def show_green(self):
        self.feed_frame.setStyleSheet("QFrame{\n"
        "background-color: qradialgradient(spread:pad, \n"
        "cx:0.5, cy:0.5, radius:0.9, fx:0.5, fy:0.5, \n"
        "stop:0 rgba(0, 0, 0, 0), stop:1 rgba(0, 200, 0, 255)); \n"
        "}")
        self.show()
        self.timer_end.start(self.time)

        # def mouseMoveEvent(self, evt):
        #     if evt.pos().y() > 900:
        #         if not self.parent.contral_ifshow:
        #             self.parent.contral.draw_show()
        #             self.parent.contral_ifshow = True
        #     else:
        #         if self.parent.contral_ifshow:
        #             self.parent.contral.draw_hide()
        #             self.parent.contral_ifshow = False
