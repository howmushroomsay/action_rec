#
# -*-------------coding=utf-8-----------------*-
'''
/* database_operation.py      9月 2021 11:01:10  
*---------------------------------------------*
*     Project Name : python_sql          
*                                             
*     File Name    : database_operation.py                  
*                                             
*     Programmer   : yufanyi@hust.edu.cn                  
*                                             
*     Start Date   : 09/07/2021  
*                                             
*     Last Update  : 09/07/2021  
*                                             
*---------------------------------------------*
*   代码描述:                                   
*   
*---------------------------------------------*
'''
#!/usr/bin/python3
import json
import pymysql

'''
{
  "host": "localhost",
  "user": "root",
  "password": "",
  "database": "prison_project_info",
  "video_model_path": "../model/video_model_parameter",
  "audio_model_path": "../model/audio_emotion_model"
}
'''
class DatabaseOperation(object):
    def __init__(self, json_path='./data/dbconfig/database_account_password.json'):
        super(DatabaseOperation, self).__init__()
        self.json_path = json_path
        database_json = self.load_json(json_path=self.json_path)
        self.db = pymysql.connect(host=database_json["host"],
                                  user=database_json["user"],
                                  password=database_json["password"],
                                  db=database_json["database"])

    def load_json(self, json_path):
        fin = open(json_path, encoding='UTF-8')
        database_json = json.load(fin)
        fin.close()
        return database_json

    def search_table(self, sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            eval_results = cursor.fetchall()
        except Exception as e:
            print(e.args)
            return
        cursor.close()
        return eval_results

    def create_table(self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
        except:
            print("Error: unable to create table")
        cursor.close()

    def insert_data(self,sql, args):
        cursor = self.db.cursor()
        try:
            cursor.executemany(sql, args)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(str(e))
        cursor.close()
    def update_data(self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(str(e))
        cursor.close()
    def close(self):
        self.db.close()


if __name__ == '__main__':
    db = DatabaseOperation(json_path='config/database_account_password.json')
    sql_search = "SELECT * FROM EVALUTION_REFERENCE WHERE COURSE_ID = %s" % (2)
    sql_insert = """INSERT INTO EMPLOYEE(FIRST_NAME,
             LAST_NAME, AGE, SEX, INCOME)
             VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
    sql_delete = "DELETE FROM EMPLOYEE WHERE AGE > %s" % (20)
    sql_update = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
    sql_search = "SELECT face_data,user_id FROM student_info"
    data = db.search_table(sql_search)
    img = data[:][0]
    user_id = data[:][1]
    for i in range(len(img)):
        sql_update = """UPDATE student_info 
                        SET face_encoding=%s
                        WHERE user_id=%s"""
        
    db.insert_update_delete_table(sql=sql_insert)

    db.close()
