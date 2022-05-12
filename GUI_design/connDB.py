import pymysql
 
# 打开数据库连接
db = pymysql.connect(host='localhost',
                     user='root',
                     password='lsm5hen1',
                     database='vue_djg')
cursor = db.cursor()
cursor.execute("select * from pjtApp_book")
results = cursor.fetchall()
for itm in results:
    print(itm)

db.close()