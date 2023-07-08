# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'g:\PoseUI\PoseUI\widgets\ui\action_followP3.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Action_FollowP3(object):
    def setupUi(self, Action_FollowP3):
        Action_FollowP3.setObjectName("Action_FollowP3")
        Action_FollowP3.resize(1920, 1080)
        self.time_frame = QtWidgets.QFrame(Action_FollowP3)
        self.time_frame.setGeometry(QtCore.QRect(0, 1000, 1920, 80))
        self.time_frame.setMinimumSize(QtCore.QSize(1920, 80))
        self.time_frame.setMaximumSize(QtCore.QSize(16777215, 80))
        self.time_frame.setSizeIncrement(QtCore.QSize(1920, 80))
        self.time_frame.setStyleSheet("QFrame{\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 rgba(0, 0, 0, 150), stop:1 rgba(100, 100, 100, 0));\n"
"}\n"
"")
        self.time_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.time_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.time_frame.setObjectName("time_frame")
        self.sld_video = QtWidgets.QSlider(self.time_frame)
        self.sld_video.setGeometry(QtCore.QRect(10, 20, 1900, 20))
        self.sld_video.setMinimumSize(QtCore.QSize(0, 20))
        self.sld_video.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.sld_video.setFont(font)
        self.sld_video.setStyleSheet("/*horizontal ：水平QSlider*/\n"
"QSlider{\n"
"    background-color: rgba(0, 0, 0,0);\n"
"}\n"
"QSlider::groove:horizontal {\n"
"border: 0px solid #bbb;\n"
"}\n"
"\n"
"/*1.滑动过的槽设计参数*/\n"
"QSlider::sub-page:horizontal {\n"
" /*槽颜色*/\n"
"background:rgb(66, 156, 227);\n"
" /*外环区域倒圆角度*/\n"
"border-radius: 2px;\n"
" /*上遮住区域高度*/\n"
"margin-top:8px;\n"
" /*下遮住区域高度*/\n"
"margin-bottom:8px;\n"
"/*width在这里无效，不写即可*/\n"
"}\n"
"\n"
"/*2.未滑动过的槽设计参数*/\n"
"QSlider::add-page:horizontal {\n"
"/*槽颜色*/\n"
"background: rgba(255,255, 255, 100);\n"
"/*外环大小0px就是不显示，默认也是0*/\n"
"border: 0px solid #777;\n"
"/*外环区域倒圆角度*/\n"
"border-radius: 2px;\n"
" /*上遮住区域高度*/\n"
"margin-top:8px;\n"
" /*下遮住区域高度*/\n"
"margin-bottom:9px;\n"
"}\n"
" \n"
"/*3.平时滑动的滑块设计参数*/\n"
"QSlider::handle:horizontal {\n"
"/*滑块颜色*/\n"
"background: rgb(193,204,208);\n"
"/*滑块的宽度*/\n"
"width: 5px;\n"
"/*滑块外环为1px，再加颜色*/\n"
"border: 1px solid rgb(193,204,208);\n"
" /*滑块外环倒圆角度*/\n"
"border-radius: 2px; \n"
" /*上遮住区域高度*/\n"
"margin-top:6px;\n"
" /*下遮住区域高度*/\n"
"margin-bottom:6px;\n"
"}\n"
"\n"
"/*4.手动拉动时显示的滑块设计参数*/\n"
"QSlider::handle:horizontal:hover {\n"
"/*滑块颜色*/\n"
"background: rgb(193,204,208);\n"
"/*滑块的宽度*/\n"
"width: 10px;\n"
"/*滑块外环为1px，再加颜色*/\n"
"border: 1px solid rgb(193,204,208);\n"
" /*滑块外环倒圆角度*/\n"
"border-radius: 5px; \n"
" /*上遮住区域高度*/\n"
"margin-top:4px;\n"
" /*下遮住区域高度*/\n"
"margin-bottom:4px;\n"
"}\n"
"")
        self.sld_video.setOrientation(QtCore.Qt.Horizontal)
        self.sld_video.setObjectName("sld_video")
        self.lab_time = QtWidgets.QLabel(self.time_frame)
        self.lab_time.setGeometry(QtCore.QRect(10, 35, 120, 25))
        self.lab_time.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lab_time.setFont(font)
        self.lab_time.setStyleSheet("color: rgba(255, 255, 255,200);\n"
"background-color: rgba(255, 255, 255,0);")
        self.lab_time.setObjectName("lab_time")

        self.retranslateUi(Action_FollowP3)
        QtCore.QMetaObject.connectSlotsByName(Action_FollowP3)

    def retranslateUi(self, Action_FollowP3):
        _translate = QtCore.QCoreApplication.translate
        Action_FollowP3.setWindowTitle(_translate("Action_FollowP3", "Form"))
        self.lab_time.setText(_translate("Action_FollowP3", "00:00:00/00:00:00"))

import res_rc
