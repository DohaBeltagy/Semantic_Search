import numpy as np
from vec_db import VecDB


PATH_DB_1M = "saved_db_1m.csv"
PATH_DB_10M = "saved_db_10m.csv"
PATH_DB_15M = "saved_db_15m.csv"
PATH_DB_20M = "saved_db_20m.csv"

db_filename_size_20M = 'saved_db_20M.dat'
db_filename_size_15M = 'saved_db_15M.dat'
db_filename_size_10M = 'saved_db_10M.dat'
db_filename_size_1M = 'saved_db_1M.dat'


database_info = {
    "1M": {
        "database_file_path": db_filename_size_1M,
        "index_file_path": PATH_DB_1M,
        "size": 106
    },
    "10M": {
        "database_file_path": db_filename_size_10M,
        "index_file_path": PATH_DB_10M,
        "size": 10 * 106
    },
    "15M": {
        "database_file_path": db_filename_size_15M,
        "index_file_path": PATH_DB_15M,
        "size": 15 * 106
    },
    "20M": {
        "database_file_path": db_filename_size_20M,
        "index_file_path": PATH_DB_20M,
        "size": 20 * 106
    }
}


db = VecDB(database_file_path = db_filename_size_1M, index_file_path = PATH_DB_1M, new_db = True, db_size = 1*(10**6))