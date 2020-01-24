# -*- coding: utf-8 -*-
#!/usr/bin/env python
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import re
import pandas as pd


def print_sysinfo(logfile, argvstr, cc='#'):
    # print(argvstr)
    date = os.popen('date').read().rstrip()
    pwd = os.popen('pwd').read().rstrip()
    hostname = os.popen('hostname').read().rstrip()
    user = os.popen('whoami').read().rstrip()

    print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(
        cc, date, user, hostname, pwd, argvstr), file=logfile)

    debug = False
    if debug:
        print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(
            cc, date, user, hostname, pwd, argvstr))


def parse_file(filepath):
    """
    Parse text at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file_object to be parsed

    Returns
    -------
    data :  Parsed data

    """

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()

        while line:
            line_pattern = r'^(\d+)\s+(\d+)\.(\S+)$'
            match_line = re.match(line_pattern, line)

            if match_line:
                sample = match_line.group(1)
                label_no = match_line.group(2)
                label_path = match_line.group(3)
                newline = label_no + "." + label_path + " " + label_no
                data.append(newline)

            line = file_object.readline()

    return data


def parse(argv):
    argvstr = " ".join(argv)

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--network", default="alexnet",
                    help="name of network")
    ap.add_argument("-p", "--prefix", default="car_ims/",
                    help="prefix of the path")
    ap.add_argument("-d", "--dataset", default="imagenet",
                    help="name of dataset")
    ap.add_argument("-l", "--logfile", default="tst.log",
                    help="name of logfile")
    ap.add_argument("-i", "--infile", default="anno.csv",
                    help="name of input file")
    ap.add_argument("-o", "--outfile", default="out.log",
                    help="name of logfile")
    ap.add_argument("-e", "--epoch", type=int, default=10,
                    help="max epoch")
    args = vars(ap.parse_args())

    outfile = open(args["outfile"], 'w', encoding='UTF-8')
    print("args: ", args)

    #df = parse_file(args["infile"])
    df = pd.read_csv(args["infile"], sep=';')
    print(df.columns)
    print(df.info())

    # Filter rows of train and test based on "test" column
    # df_train = df[df.loc[:,"test"]==0] # select test==0
    # df_test = df[df.loc[:,"test"]==1] # select test==1

    # Filter rows of train and test based on "class" column
    threshold = 98
    df_train = df[df.loc[:, "class"] <= threshold]  # select class < 98
    df_test = df[df.loc[:, "class"] > threshold]  # select class > 98

    print("\n\ndf_test: ", df_test.shape, "\n", df_test.head(), df_test.info())
    print("\n\ndf_train: ", df_train.shape,
          "\n", df_train.head(), df_train.info())

    # Filter columns of "Image" and "class"
    df_train_out = df_train[['Image', 'class']]
    df_test_out = df_test[['Image', 'class']]
    print("\n\ndf_test_out: ", df_test_out.shape,
          "\n", df_test_out.head(), df_test_out.info())
    print("\n\ndf_train_out: ", df_train_out.shape,
          "\n", df_train_out.head(), df_train_out.info())

    # add prefix on the file name
    df_train_out.loc[:, "Image"] = args["prefix"] + \
        df_train_out.loc[:, "Image"].astype(str)
    df_test_out.loc[:, "Image"] = args["prefix"] + \
        df_test_out.loc[:, "Image"].astype(str)
    print("\n\ndf_test_out: ", df_test_out.shape,
          "\n", df_test_out.head(), df_test_out.info())
    print("\n\ndf_train_out: ", df_train_out.shape,
          "\n", df_train_out.head(), df_train_out.info())

    df_train_out.to_csv('train.txt', sep=" ", header=0, index=0)
    df_test_out.to_csv('test.txt', sep=" ", header=0, index=0)

    # for row in df:
    #      print(row)

    # for row in df:
    #  print(row.rstrip())
    # for row in df:
    #  print(row.rstrip(), file=outfile)


if __name__ == "__main__":
    debug = True
    if debug:
        print(sys.argv)
        print(sys.argv[1:])  # ignore script name

    parse(sys.argv)
