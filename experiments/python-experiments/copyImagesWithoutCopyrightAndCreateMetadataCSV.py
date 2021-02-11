import os
import csv
import numpy as np
from imutils import paths

args = {}
currentdir = os.getcwd()
args["source"] = 'faces'
args["destination2"] = "faces_without_copyright_close_cut\\"

# Read csv file and create an array
file_names = []
with open("metadata/List_of_images_without_copyright.csv", newline='', encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:  # each row is a list
        file_names.append(row[1])
file_names = np.array(file_names)

# Copy files without copyright to a new folder
imagePaths = list(paths.list_images(args["source"]))
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-1]
    if name in file_names:
        command = 'copy ' + imagePath + ' ' + args["destination2"] + name
        print(i, ': ', command)
        os.popen(command)

# Create the final metadata csv file (combining two existing csv files)
name_photographer_without_copyright = []
with open("metadata/List_of_images_without_copyright.csv", newline='', encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        name_photographer_without_copyright.append(row)
name_photographer_without_copyright = np.array(name_photographer_without_copyright)
old_metadata = []
with open("metadata/metadata.csv", newline='', encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        old_metadata.append(row)
old_metadata = np.array(old_metadata)
result_file = open('metadata/metadata_new.csv', 'a', newline='', encoding='ISO-8859-1')
result_file_w = csv.writer(result_file, lineterminator=os.linesep, delimiter=';')
for (photographer, file_name) in name_photographer_without_copyright:
    row_nrs = np.where(old_metadata[: , 0] == file_name)[0]
    if len(row_nrs):
        row_nr = row_nrs[0]
        result_file_w.writerow([file_name, old_metadata[row_nr][1], old_metadata[row_nr][2], old_metadata[row_nr][3], old_metadata[row_nr][4], old_metadata[row_nr][5], old_metadata[row_nr][6], photographer, old_metadata[row_nr][8]])
result_file.close()
