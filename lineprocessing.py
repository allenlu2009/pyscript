# -*- coding: utf-8 -*-
#!/usr/bin/env python
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import re
#import pandas as pd


def print_sysinfo(logfile, argvstr, cc='#'):
  #print(argvstr)
  date = os.popen('date').read().rstrip()
  pwd  = os.popen('pwd').read().rstrip()
  hostname = os.popen('hostname').read().rstrip()
  user = os.popen('whoami').read().rstrip()
  
  print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(cc,date,user,hostname,pwd,argvstr), file=logfile)
  
  debug = False
  if debug:
    print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(cc,date,user,hostname,pwd,argvstr))

    


  
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
  ap.add_argument("-d", "--dataset", default="imagenet",
        help="name of dataset")
  ap.add_argument("-l", "--logfile", default="tst.log",
        help="name of logfile")
  ap.add_argument("-i", "--infile", default="input.log",
        help="name of input file")
  ap.add_argument("-o", "--outfile", default="out.log",
        help="name of logfile")
  ap.add_argument("-e", "--epoch", type=int, default=10,
        help="max epoch")
  args = vars(ap.parse_args())

  outfile = open(args["outfile"], 'w', encoding='UTF-8')
  print("args: ",args)

  df = parse_file(args["infile"])

  for row in df:
    print(row.rstrip())
  for row in df:
    print(row.rstrip(), file=outfile)




if __name__ =="__main__":
  debug = True
  if debug:
    print(sys.argv)
    print(sys.argv[1:]) # ignore script name

  parse(sys.argv)

