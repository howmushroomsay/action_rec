from auxiliary_tools import DatabaseOperation
import cv2
import numpy as np
# import face_recognition
def add_student(img_path, name, login_account, login_password):
    img = cv2.imread(img_path)
    img_data = np.array(cv2.imencode('.png', img)[1]).tobytes()
    face_loactions = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_loactions)
    face_encoding = np.array(face_encodings[0]).tobytes()
    print(len(face_encoding))
    db = DatabaseOperation()
    sql_insert = '''INSERT student_info
                    ( username, login_account, login_password, face_data, face_encoding)
                    VALUES(%s,%s,%s,%s,%s)'''
    args = [name, login_account, login_password, img_data, face_encoding]
    db.insert_data(sql_insert,args = [args])

if __name__ == '__main__':
    # add_student(img_path=r'C:\Users\user\Pictures\Camera Roll\llh.jpg', name='李林徽', login_account='llh', login_password='llh123456')
    db = DatabaseOperation()
    img_path = r'C:\Users\user\Desktop\t01465fcc6be9c397d7.jpg'
    img = cv2.imread(img_path)
    img_data = np.array(cv2.imencode('.png', img)[1]).tobytes()
    sql_update = """UPDATE course_standard
                    SET course_icon = %s
                    WHERE course_ID = 2
                    """
    args = [img_data]
    db.insert_data(sql_update,args = [args])