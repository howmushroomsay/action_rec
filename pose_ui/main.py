import os
import sys

from PyQt5.QtWidgets import QApplication
import multiprocessing
from widgets import *
from auxiliary_tools import DatabaseOperation

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    # win = LogMain()
    
    student_id = int(sys.argv[1])
    course_id = int(sys.argv[2])
    window_type = int(sys.argv[3])
    print(student_id, course_id, window_type)
    db = DatabaseOperation()
    # 学员id, 课程id, 训练类型
    if window_type == 0:
        win = Reaction_Train_Main(None, db=db, course_id=course_id, student_id=student_id)
    elif window_type == 1:
        win = Action_Follow_Main(None, db=db, action_id=course_id)
    elif window_type == 2:
        win = Action_Eval_Main(None, db=db, action_id=course_id, student_id=student_id)
    
    # win = Reaction_Train_Main(0, db, 0)
    # camera = "rtsp://admin:nvidia001@192.168.1.64/Streaming/Channels/2"
    # camera = 0
    # win = Action_Eval_Main(0, 0, 32, 0, camera=camera)
    # win = Action_Follow_Main(0, db, 22)
    win.show()
    code = app.exec_()
    course_dirs = ['./data/course_action/response', './data/course_action/std']
    for course_dir in course_dirs:
        for file in os.listdir(course_dir):
            path_ = os.path.join(course_dir, file)
            if os.path.isdir(path_):
                for file_ in os.listdir(path_):
                    try:
                        os.remove(os.path.join(path_, file_))
                    except:
                        continue
            else:
                try:
                    os.remove(path_)
                except:
                    continue
    sys.exit(code)
