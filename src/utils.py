import os


def create_directories(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)