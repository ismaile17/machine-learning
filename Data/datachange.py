"""import csv

with open('predictive_maintenance.csv', 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    for row in reader:
        with open("data2.txt", "a") as myfile:
            myfile.write(row[0]+"\n")

"""

import csv
import zlib

with open('data2.txt', 'r') as in_file:
    lines = in_file.read().splitlines()
    stripped = [line.split(",") for line in lines]
    grouped = zip(*[stripped]*1)
    with open('data2.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for group in grouped:
            writer.writerows(group)