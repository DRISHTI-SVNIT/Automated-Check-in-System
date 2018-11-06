import psycopg2

try :
    conn = psycopg2.connect("dbname='College_Students' user='postgres' host='localhost' password='postgres'")
    
except :
    print("I am unable to connect to the database")

cur = conn.cursor()
drawing = open('croped1.png', 'rb').read()
#cur.execute("CREATE TABLE College_Students_2(Name text,admission_no text)")
'''
cur.execute("INSERT INTO t1(name,uid,pic) "+"VALUES(%s,%s,%s)",('Atharva', 'u16co063', psycopg2.Binary(drawing)))
conn.commit()
cur.close()
'''
cur.execute(""" SELECT * FROM t1 """)
blob = cur.fetchone()
open('test/'+ 'hello' + '.' + 'jpg', 'wb').write(blob[2])
cur.close()