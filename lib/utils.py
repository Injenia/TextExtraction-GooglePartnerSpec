# This Python file uses the following encoding: utf-8
from __future__ import division

from google.cloud import storage
from shutil import copyfile, rmtree
from pathlib2 import Path

import os
import glob
import json
import pickle
import csv
import zipfile

# lists
def batch_list(l, n):
    return [l[i:i+n] for i in range(0,len(l),n)]

# Files
def ensure_dir_exists(directory):
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_files_by_extensions(directory, extensions):
    res = []
    for extension in extensions:
        res.extend(glob.glob(os.path.join(directory, "*." + extension)))
    return res


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and not name.startswith(".")]


def rm_dir(dir):
    os.path.exists(dir) and rmtree(dir)
    
def move_flattened_files(src_dir, out_dir, filt):
    p = Path(src_dir)
    if p.is_dir():
        for f in p.iterdir():
            if f.is_file() and filt(f.absolute()):
                copyfile(str(f.absolute()),str(out_dir + u'/' + f.name))
            else:
                move_flattened_files(str(f.absolute()), out_dir, filt)

def list_dir_full(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

#JSON
def load_json(path, verbose=False):
    if os.path.isfile(path):
        if (verbose):
            print("opening" + " " + os.path.abspath(path))
        result = json.load(open(path, "r"))
        if (verbose):
            print("json retreived succesfully")
        return result
    else:
        raise Exception(os.path.abspath(path) + " does not exists")


def save_json(dictionary, path, verbose=False):
    ensure_dir_exists(path)
    if (verbose):
        print('saving ' + os.path.abspath(path) + " ...")
    json.dump(dictionary, open(path, 'w'), sort_keys=True, indent=4)
    if (verbose):
        print(os.path.abspath(path) + " saved successfully")


#Pickle
def load_pickle(path, verbose=False):
    if os.path.isfile(path):
        if (verbose):
            print("opening" + " " + os.path.abspath(path))
        result = pickle.load(open(path, "r"))
        if (verbose):
            print("pickle retreived succesfully")
        return result
    else:
        raise Exception(os.path.abspath(path) + " does not exists")


def save_pickle(dictionary, path, verbose=False):
    ensure_dir_exists(path)
    if (verbose):
        print('saving ' + os.path.abspath(path) + " ...")
    pickle.dump(dictionary, open(path, 'w'))
    if (verbose):
        print(os.path.abspath(path) + " saved successfully")


#CSV
def load_csv(path, verbose=False):
    res = []
    if(verbose):
        print('opening '+os.path.abspath(path))
    with open(path, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            res.append(row)
    if(verbose):
        print("csv content retreived succesfully")
    return res

def read_csv_from_string(content, verbose=False, delimiter=None, quotechar=None):
    res = []
    if delimiter is None and quotechar is None:
        csvreader = csv.reader(content)
    else:
        csvreader = csv.reader(content, delimiter=delimiter, quotechar=quotechar)
    for row in csvreader:
        res.append(row)
    if(verbose):
        print("csv content retreived succesfully")
    return res

def save_csv(content, path, verbose=False):
    ensure_dir_exists(path)
    if(verbose):
        print('saving '+os.path.abspath(path)+" ...")
    with open(path, 'w') as outfile:
        for c in content:
            outfile.write(','.join([str(x).replace(",","") for x in c])+'\n')
    if(verbose):
        print(os.path.abspath(path)+" saved successfully")


#Download file
def download_file_asynchronously(url, destination, new_name=None):
    dest_directory = destination
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = url.split('/')[-1]
    if new_name != None:
        filename = new_name
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def download_file_synchronously(url, destination, new_name=None, verbose=False):
    dest_directory = destination
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = url.split('/')[-1]
    if new_name != None:
        filename = new_name
    filepath = os.path.join(dest_directory, filename)
    # if not os.path.exists(filepath):
    if (verbose):
        print('downloading ' + url + " to " + filepath)
    f = urllib.request.urlopen(url)
    with open(filepath, "w") as imgFile:
        imgFile.write(f.read())
    return filepath


#Unzip
def unzip_file(path, destination, verbose=False):
    if(verbose):
        print("unzipping "+path+" to "+destination)
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall(destination)
    zip_ref.close()
    if(verbose):
        print("done.")


#Progress logger
class ProgressLogger:
    def __init__(self, total, percent_step):
        self.total = total
        self.percent_step = percent_step
        self.actual = 0
        self.percent_actual = 0
        self.next_log = self.percent_step

    def update(self):
        result = False
        new_actual = self.actual + 1
        new_percent_actual = 100 * new_actual / self.total
        if self.percent_actual <= self.next_log <= new_percent_actual:
            print(str(new_actual) + "/" + str(self.total) + " - " + str(new_percent_actual) + "%")
            self.next_log += self.percent_step
            result = True
        self.actual = new_actual
        self.percent_actual = new_percent_actual
        return result
    
    
#Google Cloud Storage
def read_gcs_file(project,path):
    splitted = path.split('/')
    bucket = splitted[2]
    filepath = '/'.join(splitted[3:])
    
    client = storage.Client(project=project)
    bucket = storage.bucket.Bucket(client,name=bucket)
    blob = storage.blob.Blob(filepath,bucket)
    return blob.download_as_string()


def write_gcs_file(project,path,content):
    splitted = path.split('/')
    bucket = splitted[2]
    filepath = '/'.join(splitted[3:])
    client = storage.Client(project=project)
    bucket = storage.bucket.Bucket(client,name=bucket)
    blob = storage.blob.Blob(filepath,bucket)
    blob.upload_from_string(content)   
    