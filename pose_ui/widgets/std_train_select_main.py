import sys

from PyQt5.QtWidgets import QApplication,QWidget
from PyQt5 import QtCore

from .ui import Ui_Std_Train_Select
from .action_follow_main import Action_Follow_Main
from .action_eval_main import Action_Eval_Main
class Std_Train_Select_Main(QWidget,Ui_Std_Train_Select):

    def __init__(self, parent, db, course_id, action_id, student_id, using_sensor, camera):
        super(Std_Train_Select_Main, self).__init__()
        self.setupUi(self)
        self.windowinit()

        self.parent = parent
        self.db = db
        self.course_id = course_id
        self.action_id = action_id
        self.student_id = student_id
        self.using_sensor = using_sensor
        self.camera = camera
        self.initfun()

    def initfun(self):
        self.btn_ac_analysis.clicked.connect(self.openNextWindow)
        self.btn_ac_train.clicked.connect(self.openNextWindow)
        self.btn_back.clicked.connect(self.back)
        sql_search = """SELECT action_Name
                        FROM action_info
                        WHERE action_ID={}""".format(self.action_id)
        data =self.db.search_table(sql_search)
        self.lab_name.setText(data[0][0])
    def openNextWindow(self):
        if self.sender().text() == '动作分析':
            self.nextWindow = Action_Follow_Main(self, self.db, self.action_id)
        else:
            self.nextWindow = Action_Eval_Main(self, self.db, self.course_id, self.action_id, self.student_id, self.using_sensor, self.camera)
        self.nextWindow.show()
        self.hide()
    def back(self):
        # 回到动作选择界面
        self.close()
        self.parent.show()
    
    def windowinit(self):
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Std_Train_Select_Main()
    win.show()
    sys.exit(app.exec_())
