import psycopg2

class Connect:

    def push(self,f_name,l_name,ad_no):
        try :
            conn = psycopg2.connect("dbname='College_Students' user='postgres' host='localhost' password='postgres'")
            
        except :
            print("I am unable to connect to the database")

        cur = conn.cursor()
        drawing = open('croped_temp.png', 'rb').read()
        #cur.execute("CREATE TABLE College_Students_2(Name text,admission_no text)")
        
        cur.execute("INSERT INTO t1(name,uid,pic) "+"VALUES(%s,%s,%s)",(f_name, l_name, psycopg2.Binary(drawing)))
        conn.commit()
        cur.close()
        '''
        cur.execute(""" SELECT * FROM t1 """)
        blob = cur.fetchone()
        open('test/'+ 'hello' + '.' + 'jpg', 'wb').write(blob[2])
        cur.close()
        '''