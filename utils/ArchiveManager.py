import datetime
import time
import os
import logging


def get_date():
    date_today = datetime.datetime.now().strftime("%y%m%d")
    return date_today


def get_time():
    date_today = datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S")
    return date_today


class ArchiveManager:

    def __init__(self, problem_name, exp_name):
        self.problem_name = problem_name
        self.exp_name = exp_name
        self.time_stamp = str(int(time.time()))

    def get_tag(self):
        tag = self.problem_name + "_" + self.exp_name
        return tag

    def get_tag_dir(self):
        path = os.path.join(".", "runs", self.get_tag(), "")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_exp_dir(self):
        date_today = get_date()
        path = os.path.join(self.get_tag_dir(), date_today + "_" + self.time_stamp, "")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_exp_log_path(self):
        return os.path.join(self.get_exp_dir(), "log.txt")


if __name__ == '__main__':
    am = ArchiveManager("ML", "test")
    print(am.get_tag())
    print(am.get_tag_dir())
    print(am.get_exp_dir())
    print(am.get_exp_log_path())
