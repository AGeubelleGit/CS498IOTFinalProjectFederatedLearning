import mysql.connector
import csv
def connect_to_db():
    db = mysql.connector.connect(host="localhost", user="user", password="password", database="final_project") 
    return db

def insert_row(db, row): # row should be a tuple, with the values in order of time, employee_name, employee_id
    cursor = db.cursor()

    sql = "INSERT INTO employee_data (time_val, employee_name, employee_id) VALUES (%s,%s,%s)"
    cursor.execute(sql, row)
    db.commit()

def get_rows(db): # get the rows and writes them to a csv file
    cursor = db.cursor()
    cursor.execute("SELECT * from employee_data")
    field_names = [item[0] for item in cursor.description]
    rows=cursor.fetchall()

    with open("data.csv","w") as f:
        file = csv.writer(f)
        file.writerow(field_names)
        file.writerows(rows)
        
db = connect_to_db()
insert_row(db, (0, "alex", 1))
get_rows(db)