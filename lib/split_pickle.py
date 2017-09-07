# This Python file uses the following encoding: utf-8
import pickle
import argparse
from os import path

parser = argparse.ArgumentParser()

parser.add_argument(
     '--picklefile',
     type=str,
     action='append',
     required=True,
     help='The path to the pickle file to be split'
     )

parser.add_argument(
     '--destfolder',
     type=str,
     action='append',
     required=True,
     help='The path to the output folder'
     )

parser.add_argument(
     '--numdocs',
     type=int,
     action='append',
     required=True,
     help='The number of docs for the split'
     )


args, _ = parser.parse_known_args()

filename = args.picklefile[0]
destfolder = args.destfolder[0]
numdocs = args.numdocs[0]

def split_pickles(filename, dest, n_docs):
    with open(filename, 'rb') as f:
        data, labels = pickle.load(f)
    print "file loaded"
    j = 0
    i = 0
    while j < len(data):
        with open(path.join(dest,filename[:-2])+str(i)+'.p','wb') as f:
            pickle.dump([data[j:j+n_docs],labels[j:j+n_docs]], f)
        j += n_docs
        print i
        i += 1

split_pickles(filename, destfolder, numdocs)

