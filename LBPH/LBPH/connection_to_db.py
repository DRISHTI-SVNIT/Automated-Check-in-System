import psycopg2
import numpy as np
import cv2
class Connect:

    def push(self,f_name,l_name,ad_no):
        try :
            conn = psycopg2.connect("dbname='College_Students' user='postgres' host='localhost' password='postgres'")
            
        except :
            print("I am unable to connect to the database")

        cur = conn.cursor()
        drawing = open('croped_temp.png', 'rb').read()
        #cur.execute("CREATE TABLE College_Students_2(Name text,admission_no text)")
        
        cur.execute("INSERT INTO t1(name,uid,pic,l_name) "+"VALUES(%s,%s,%s,%s)",(f_name, ad_no, psycopg2.Binary(drawing),l_name))
        conn.commit()
        cur.close()
        '''
        cur.execute(""" SELECT * FROM t1 """)
        blob = cur.fetchone()
        open('test/'+ 'hello' + '.' + 'jpg', 'wb').write(blob[2])
        cur.close()
        '''

    def fetch(self):
        try :
            conn = psycopg2.connect("dbname='College_Students' user='postgres' host='localhost' password='postgres'")
            
        except :
            print("I am unable to connect to the database")

        cur = conn.cursor()
        f_name=[]
        l_name=[]
        ad_no=[]
        faces=[]
        labels=[]
        cur.execute(""" SELECT * FROM t1 """)
        for i in range(0,cur.rowcount):
            blob = cur.fetchone()
            open('test/'+ str(i) + '.jpg', 'wb').write(blob[2])
            img=cv2.imread('test/'+str(i)+'.jpg')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces.append(img)
            labels.append(int(i))
            f_name.append(str(blob[0]))
            l_name.append(str(blob[3]))
            ad_no.append(str(blob[1]))
        cur.close()
        return faces,labels,f_name,l_name,ad_no

    