-- Run the following command to set up the schema:
--     mysql -u user -p < create_schema.sql 
--     This deletes the table and recreates it empty
CREATE DATABASE IF NOT EXISTS final_project;
USE final_project;

DROP TABLE IF EXISTS employee_data;

CREATE TABLE employee_data (
	id int KEY NOT NULL AUTO_INCREMENT,
	time_val INTEGER,	
	employee_name VARCHAR(255), 
    	employee_id INTEGER
);