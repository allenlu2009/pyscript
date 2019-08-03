# -*- coding: utf-8 -*-
#!/usr/bin/env python
import argparse
import sys
import os
import re

def print_sysinfo(logfile, argvstr, cc='#'):
  print(argvstr)
  date = os.popen('date').read().rstrip()
  pwd  = os.popen('pwd').read().rstrip()
  hostname = os.popen('hostname').read().rstrip()
  user = os.popen('whoami').read().rstrip()
  
  print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(cc,date,user,hostname,pwd,argvstr), file=logfile)
  
  debug = False
  if debug:
    print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(cc,date,user,hostname,pwd,argvstr))



def main(argv):
  argvstr = " ".join(argv)
  
  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-n", "--network", default="alexnet",
        help="name of network")
  ap.add_argument("-d", "--dataset", default="imagenet",
        help="name of dataset")
  ap.add_argument("-l", "--logfile", default="test.log",
        help="name of logfile")
  ap.add_argument("-o", "--outfile", default="out.log",
        help="name of logfile")
  ap.add_argument("-e", "--epoch", type=int, default=10,
        help="max epoch")
  args = vars(ap.parse_args())

  outfile = open(args["outfile"], 'w', encoding='UTF-8')
  print("args: ",args)
  print_sysinfo(outfile, argvstr, '//')

  START = 'rate   iter_k   epoch  num_m | valid_loss/acc | train_loss/acc | batch_loss/acc |  time'
  END = '========='

  rows = open(args["logfile"], 'r').read()
  result = re.findall('%s(.*)%s' % (START,END), rows)
  print("result: ", result)
  

if __name__ =="__main__":
  debug = True
  if debug:
    print(sys.argv)
    print(sys.argv[1:]) # ignore script name

  main(sys.argv)

