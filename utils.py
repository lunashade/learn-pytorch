"""Utility Class."""
import csv
import os


class CSVWriter(object):
    def __init__(self, filename, *args):
        self.filename = filename
        self.header = args
        if not os.path.isfile(filename):
            with open(filename, 'w') as fp:
                writer = csv.writer(fp)
                writer.writerow(self.header)

    def write(self, **kwargs):
        row = []
        for column in self.header:
            if column in kwargs:
                row.append(kwargs[column])
            else:
                row.append(None)
        with open(self.filename, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(row)
