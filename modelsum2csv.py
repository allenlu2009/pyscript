# -*- coding: utf-8 -*-
#!/usr/bin/env python
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import re
import pandas as pd

# set up regular expressions
# use https://regexper.com to visualise these if required
rx_dict = {
    'in':  re.compile(r'^\s*(?P<layer>Input\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'conv2d': re.compile(r'^\s*(?P<layer>Conv2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'bn2d': re.compile(r'^\s*(?P<layer>BatchNorm2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'relu': re.compile(r'^\s*(?P<layer>ReLU\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'linear': re.compile(r'^\s*(?P<layer>Linear\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'pool2d': re.compile(r'^\s*(?P<layer>\S*Pool2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'bottleneck': re.compile(r'^\s*(?P<layer>Bottleneck\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'drop2d': re.compile(r'^\s*(?P<layer>Dropout2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
}

model_dict = {
    'in':  re.compile(r'^\s*(?P<layer>Input\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'conv2d': re.compile(r'^\s*(?P<layer>Conv2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'bn2d': re.compile(r'^\s*(?P<layer>BatchNorm2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'relu': re.compile(r'^\s*(?P<layer>ReLU\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'linear': re.compile(r'^\s*(?P<layer>Linear\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'pool2d': re.compile(r'^\s*(?P<layer>\S*Pool2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'bottleneck': re.compile(r'^\s*(?P<layer>Bottleneck\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'drop2d': re.compile(r'^\s*(?P<layer>Dropout2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
}



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

    
def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def _parse_model(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None



  
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
    net = []
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()

        while line:
          # at each line check for a match with a regex parsing output shape
          key, match = _parse_line(line)

          if key == 'in':
                layer = match.group('layer')
                param = int(match.group('param').replace(',', ''))  # remove comma
                shape_arr = match.group('shape').split(', ')
                batch = int(shape_arr[0])
                channel = int(shape_arr[1])
                width = int(shape_arr[2])
                height = int(shape_arr[3])
                row = {
                      'Layer': layer,
                      'Batch': batch,
                      'Channel': channel,
                      'Width': width,
                      'Height': height,
                      'Param': param
                }
                data.append(row)
                print('{} \t\t[{}, {}, {}, {}]\t{}'.format(layer, batch, channel, width, height, param))

          if key == 'conv2d':
                layer = match.group('layer')
                param = int(match.group('param').replace(',', ''))  # remove comma
                shape_arr = match.group('shape').split(', ')
                batch = int(shape_arr[0])
                channel = int(shape_arr[1])
                width = int(shape_arr[2])
                height = int(shape_arr[3])
                row = {
                      'Layer': layer,
                      'Batch': batch,
                      'Channel': channel,
                      'Width': width,
                      'Height': height,
                      'Param': param
                }
                data.append(row)
                print('{} \t\t[{}, {}, {}, {}]\t{}'.format(layer, batch, channel, width, height, param))

          if key in ('bn2d', 'relu', 'pool2d', 'bottleneck'):
                layer = match.group('layer')
                param = int(match.group('param').replace(',', ''))  # remove comma
                shape_arr = match.group('shape').split(',')
                batch = int(shape_arr[0])
                channel = int(shape_arr[1])
                width = int(shape_arr[2])
                height = int(shape_arr[3])
                row = {
                      'Layer': layer,
                      'Batch': batch,
                      'Channel': channel,
                      'Width': width,
                      'Height': height,
                      'Param': param
                }
                data.append(row)
                print('{} \t\t[{}, {}, {}, {}]\t{}'.format(layer, batch, channel, width, height, param))

          if key == 'drop2d':
                layer = match.group('layer')
                param = int(match.group('param').replace(',', ''))  # remove comma
                shape_arr = match.group('shape').split(',')
                batch = int(shape_arr[0])
                channel = int(shape_arr[1])
                width = int(shape_arr[2])
                height = int(shape_arr[3])
                row = {
                      'Layer': layer,
                      'Batch': batch,
                      'Channel': channel,
                      'Width': width,
                      'Height': height,
                      'Param': param
                }
                data.append(row)
                print('{} \t\t[{}, {}, {}, {}]\t{}'.format(layer, batch, channel, width, height, param))

          if key == 'linear':
                layer = match.group('layer')
                param = int(match.group('param').replace(',', ''))  # remove comma
                shape_arr = match.group('shape').split(',')
                batch = int(shape_arr[0])
                channel = int(shape_arr[1])
                row = {
                      'Layer': layer,
                      'Batch': batch,
                      'Channel': channel,
                      'Param': param
                }
                data.append(row)
                print('{} \t\t[{}, {}]\t{}'.format(layer, batch, channel, param))

          # at each line check for a match with a regex parsing output shape
          
          #key, match = _parse_model(line)

          #if key == 'in':

          line = file_object.readline()
          
        # create a pandas DataFrame from the list of dicts
        data = pd.DataFrame(data, columns = ["Layer", "Channel", "Width", "Height", "Param"]) 
        print(data)
        #data.set_index('Layer', inplace=True)
        # consolidate df to remove nans
        #data = data.groupby(level=data.index.names).first()
        # upgrade Score from float to integer
        #data = data.apply(pd.to_numeric, errors='ignore')
    
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
  ap.add_argument("-i", "--infile", default="model.log",
        help="name of input file")
  ap.add_argument("-o", "--outfile", default="out.log",
        help="name of logfile")
  ap.add_argument("-e", "--epoch", type=int, default=10,
        help="max epoch")
  args = vars(ap.parse_args())

  outfile = open(args["outfile"], 'w', encoding='UTF-8')
  print("args: ",args)

  data = parse_file(args["infile"])
  
  print(data)
  

  #for row in df:
  #  print(row.rstrip())
  #for row in df:
  #  print(row.rstrip(), file=outfile)




if __name__ =="__main__":
  debug = True
  if debug:
    print(sys.argv)
    print(sys.argv[1:]) # ignore script name

  parse(sys.argv)

