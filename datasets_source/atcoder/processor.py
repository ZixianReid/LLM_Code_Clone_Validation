import os
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
            specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    connection = None
    try:
        connection = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return connection


def create_database_if_not_exists(db_file):
    # Check if file exists
    if os.path.exists(db_file):
        print(f"Database {db_file} already exists.")
    else:
        # Connect - this will create database if not exists
        print(f"Database {db_file} doesn't exist. Creating...")
        conn = create_connection(db_file)
        if conn:
            conn.close()
            print(f"Database {db_file} created successfully.")
        else:
            print(f"Failed to create database {db_file}.")


if __name__ == '__main__':
    create_connection("/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/atcoder/java-python-clones.db")
    """SELECT
   s1.id AS id_anchor,
   s2.id AS id_positive,
   s1.source AS source_anchor,
   s2.source AS source_positive,
   s1.language_code AS language_code_anchor,
   s2.language_code AS language_code_positive
FROM
   samples sp
   INNER JOIN submissions s1 ON sp.anchor_id = s1.id
   INNER JOIN submissions s2 ON sp.positive_id = s2.id
   """