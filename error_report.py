#!/usr/bin/python

# error_report.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import datetime
import traceback
import sys
import os


def get_error_report():
    error_report = ErrorReport()
    return error_report


class ErrorReport():        

    def __init__(self):
        return

    def start_log(self):
        timestamp = str(datetime.datetime.now())
        file_name = "Log_" + timestamp + ".txt"
        self.log_file = open(file_name, 'w')

    def end_log(self):
        self.log_file.close()

    def write_error(self):
        traceback.print_exc(file=self.log_file)
        self.log_file.write("\n")
        self.log_file.flush()
        os.fsync(self.log_file)

    def write_message(self, message=""):
        self.log_file.write(message)
        self.log_file.write("\n\n")
        self.log_file.flush()
        os.fsync(self.log_file)
