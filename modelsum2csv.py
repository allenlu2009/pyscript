# -*- coding: utf-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import re
import pandas as pd

# enable logging for debug
import logging

#logger = logging.getLogger('modelsum2csv')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set logging default level DEBUG
# set log format
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

# set up regular expressions
# use https://regexper.com to visualise these if required
rx_dict = {
    'in':  re.compile(r'^\s*(?P<layer>Input)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'conv2d': re.compile(r'^\s*(?P<layer>Conv2d)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'bn2d': re.compile(r'^\s*(?P<layer>BatchNorm2d)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'relu': re.compile(r'^\s*(?P<layer>ReLU)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'linear': re.compile(r'^\s*(?P<layer>Linear)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'pool2d': re.compile(r'^\s*(?P<layer>\S*Pool2d)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'bottleneck': re.compile(r'^\s*(?P<layer>Bottleneck)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
    'drop2d': re.compile(r'^\s*(?P<layer>Dropout2d)\S+\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
}

model_dict = {
    'conv2d': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>Conv2d)\((?P<arg>.*)\)'),
    'bn2d': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>BatchNorm2d)\((?P<arg>.*)\)'),
    'maxpool2d': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>MaxPool2d)\((?P<arg>.*)\)'),
    'avgpool2d': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>AdaptiveAvgPool2d)\((?P<arg>.*)\)'),
    'linear': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>Linear)\((?P<arg>.*)\)'),
    'relu': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>ReLU)\((?P<arg>.*)\)'),
    'bottleneck': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>Bottleneck)\((?P<arg>.*)'),
    'seq': re.compile(r'^\s*\(\S+\)\:\s*(?P<layer>Sequential)\((?P<arg>.*)'),
    'drop2d': re.compile(r'^\s*(?P<layer>Dropout2d\S+)\s+\[(?P<shape>.+)\]\s+(?P<param>\S+)$'),
}


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


def _parse_line(line):

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def _parse_line_shape(line):

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def _parse_line_model(line):

    for key, rx in model_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def parse_file_2(filepath):
    try:
        # open the file and read through it line by line
        with open(filepath, 'r') as fileObject:
            fileLst = fileObject.read().split("\n")

        fileLstClean = [data.strip() for data in fileLst]

        modelStartIndex = fileLstClean.index('=Model Start=')
        modelStopIndex = fileLstClean.index('=Model End=')

        shapeStartIndex = fileLstClean.index('=Shape Start=')
        shapeStopIndex = fileLstClean.index('=Shape End=')

        #modelStr = "\n".join(fileLst[modelStartIndex+1:modelStopIndex])
        #shapeStr = "\n".join(fileLst[shapeStartIndex+1:shapeStopIndex])
        # print(shapeStr)
        # print(modelStr)

        modelLst = fileLst[modelStartIndex+1:modelStopIndex]
        shapeLst = fileLst[shapeStartIndex+1:shapeStopIndex]

        # create DataFrame for output shape of model given input size
        dfShape = parse_shape(shapeLst)
        # create DataFrame for model based on kernel size, etc.
        dfModel = parse_model(modelLst)

        # dfShape and dfModel
        dfMerge = merge_shape_model(dfShape, dfModel)

        logger.debug("dfMerge = \n %s" % (dfMerge))

        # plot parameter vs. layer
        dfPlot(dfMerge)

    except Exception as e:
        print(e)


def parse_shape(shapeLst):

    data = []

    for line in shapeLst:
        key, match = _parse_line_shape(line)

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
                'Width': 1,
                'Height': 1,
                'Param': param
            }
            data.append(row)

    dfShape = pd.DataFrame(
        data, columns=["Layer", "Channel", "Width", "Height", "Param"])
    dfShape.set_index(["Layer"], inplace=True)
    # print(dfShape)
    return dfShape


# use line + regex to parse the model summary. Need to rewrite it using parser
def parse_model(modelLst):

    #modelName = modelLst[0]
    data = []

    for line in modelLst:
        key, match = _parse_line_model(line)

        if key == 'conv2d':
            layer = match.group('layer')
            arg = match.group('arg').replace(" ", "")  # remove space for regex
            # further parsing the arg
            pat = r'(\d+),(\d+),.*kernel.*=\((\d+),(\d+)\),.*stride=\((\d+),(\d+)\)'
            arg_m = re.match(pat, arg)
            if arg_m:
                row = {
                    'Layer': layer,
                    'ChI': int(arg_m.group(1)),
                    'ChO': int(arg_m.group(2)),
                    'Kernel': int(arg_m.group(3)),
                    'Stride': int(arg_m.group(5)),
                    'Padding': -1
                }
                data.append(row)

        if key == 'maxpool2d':
            layer = match.group('layer')
            arg = match.group('arg').replace(" ", "")  # remove space for regex
            # further parsing the arg
            pat = r'.*kernel.*=(\d+),.*stride=(\d+),.*padding=(\d+).*'
            arg_m = re.match(pat, arg)
            if arg_m:
                row = {
                    'Layer': layer,
                    'ChI': -1,
                    'ChO': -1,
                    'Kernel': int(arg_m.group(1)),
                    'Stride': int(arg_m.group(2)),
                    'Padding': int(arg_m.group(3))
                }
                data.append(row)

        if key == 'linear':
            layer = match.group('layer')
            arg = match.group('arg').replace(" ", "")  # remove space for regex
            # further parsing the arg
            pat = r'.*in_features=(\d+),.*out_features=(\d+),.'
            arg_m = re.match(pat, arg)
            if arg_m:
                row = {
                    'Layer': layer,
                    'ChI': int(arg_m.group(1)),
                    'ChO': int(arg_m.group(2)),
                    'Kernel': -1,
                    'Stride': -1,
                    'Padding': -1
                }
                data.append(row)

        if key == 'bn2d':
            layer = match.group('layer')
            arg = match.group('arg').replace(" ", "")  # remove space for regex
            # further parsing the arg
            pat = r'(\d+),.*'
            arg_m = re.match(pat, arg)
            if arg_m:
                row = {
                    'Layer': layer,
                    'ChI': int(arg_m.group(1)),
                    'ChO': int(arg_m.group(1)),
                    'Kernel': -1,
                    'Stride': -1,
                    'Padding': -1
                }
                data.append(row)

        if key in ['relu', 'bottleneck', 'seq']:
            layer = match.group('layer')
            arg = match.group('arg').replace(" ", "")  # remove space for regex
            row = {
                'Layer': layer,
                'ChI': -1,
                'ChO': -1,
                'Kernel': -1,
                'Stride': -1,
                'Padding': -1
            }
            data.append(row)

        if key == 'avgpool2d':
            layer = match.group('layer')
            arg = match.group('arg').replace(" ", "")  # remove space for regex
            # further parsing the arg
            pat = r'.*output_size=\((\d+),(\d+)\)'
            arg_m = re.match(pat, arg)
            if arg_m:
                row = {
                    'Layer': layer,
                    'ChI': -1,
                    'ChO': -1,
                    'Kernel': int(arg_m.group(1)),
                    'Stride': int(arg_m.group(1)),
                    'Padding': -1
                }
                data.append(row)

    dfModel = pd.DataFrame(
        data, columns=["Layer", "ChI", "ChO", "Kernel", "Stride", "Padding"])
    dfModel.set_index(["Layer"], inplace=True)
    #dfMOdel = df.apply(pd.to_numeric, errors='ignore')
    #print(dfModel[dfModel.index == 'Conv2d'])
    #print(dfModel[dfModel.index == 'Conv2d'].count())
    # print(dfModel)
    # print(dfModel.count())
    return dfModel


def match_shape_model(dfShape, dfModel):
    # debug
    logger.debug("dfShape count: %s" % (dfShape.count()))
    logger.debug("dfModel count: %s" % (dfModel.count()))

    # convert the groupby 'Layer' of dfShape and dfModel to dict for comparison
    out1 = dict(dfShape.groupby(['Layer']).size())
    out2 = dict(dfModel.groupby(['Layer']).size())
    # debug
    # print(len(out1))
    # print(len(out2))

    # find the shared items
    #shared_items = {k: out1[k] for k in out1 if k in out2 and out1[k] == out2[k]}
    shareItems = out1.items() & out2.items()
    diffItems = out2.items() ^ out1.items()

    diffItemsLst = []
    for term in diffItems:
        diffItemsLst.append(term[0])
        #logger.debug("different layer: %s" %(term[0]))
    logger.debug("different layer: %s" % (diffItemsLst))
    diffItemsLstUnique = list(set(diffItemsLst))
    diffItemsLstUnique.append('Bottleneck')
    #shareItemsLst = list(shareItems.keys())
    #diffItemsLst = list(diffItems.keys())
    # print(diffItemsLst)

    shapeDiff = out1.keys() - out2.keys()  # key in A but not in B
    modelDiff = out2.keys() - out1.keys()  # key in B but not in A
    logger.debug("shareItems: %s" % (shareItems))
    logger.debug("diffItems: %s" % (diffItems))
    logger.debug("dfShape layer size: %s\n" %
                 (dfShape.groupby(['Layer']).size()))
    logger.debug("dfModel layer size: %s\n" %
                 (dfModel.groupby(['Layer']).size()))

    # remove layer in the diffItemsLst
    dfShapeClean = dfShape[~dfShape.index.isin(diffItemsLstUnique)]
    dfModelClean = dfModel[~dfModel.index.isin(diffItemsLstUnique)]

    #print(dfShapeClean[1:10], dfShape[1:10])
    #print(dfModelClean, dfModel)
    # print(dfShapeClean.index[0:10])
    # print(dfModelClean.index[0:10])

    # chck if the two dataframe match
    match = dfShapeClean.index.equals(dfModelClean.index)

    return match, dfShapeClean, dfModelClean


def merge_shape_model(dfShape, dfModel):

    # debug: compute how many layers for each layer type
    # print(dfShape.groupby(['Layer']).size())
    # print(dfModel.groupby(['Layer']).size())

    # align the size of dfShape and dfModel
    match, dfShapeClean, dfModelClean = match_shape_model(dfShape, dfModel)

    # print(dfShapeClean[0:10])
    # print(dfModelClean[0:10])
    # print(dfMerge[0:10])

    if match:
        # Concat two dataframe when matched!
        dfMerge = pd.concat([dfShapeClean, dfModelClean], axis=1)

        # reset index to get the row number
        dfMerge.reset_index(inplace=True)
        # print(dfMerge)
    else:
        print("Shape and Model NOT Match, Please Check Again!")
        exit(0)

    # start to clean the date
    # 1. Copy Width and Height to OFmapX and OFmapY
    dfMerge['OFmapX'] = dfMerge['Width']
    dfMerge['OFmapY'] = dfMerge['Height']
    dfMerge['IFmapX'] = 0
    dfMerge['IFmapY'] = 0
    dfMerge['ParamW'] = 0
    dfMerge['ParamIF'] = 0
    dfMerge['ParamOF'] = 0
    dfMerge['MAC'] = 0
    dfMerge['MAC2W'] = 0
    dfMerge['MAC2IF'] = 0
    dfMerge['MAC2OF'] = 0
    # print(dfMerge)

    # 2. Find CHI,CHO=-1 to replace with Channel, fixed later
    for index, row in dfMerge.iterrows():
        if index == 0:  # put input layer as IFmap
            dfMerge.loc[index, 'IFmapX'] = 224
            dfMerge.loc[index, 'IFmapY'] = 224
        else:
            dfMerge.loc[index, 'IFmapX'] = dfMerge.loc[index-1, 'OFmapX']
            dfMerge.loc[index, 'IFmapY'] = dfMerge.loc[index-1, 'OFmapY']

        if row['ChI'] == -1:
            dfMerge.loc[index, 'ChI'] = row['Channel']
        if row['ChO'] == -1:
            dfMerge.loc[index, 'ChO'] = row['Channel']

    for index, row in dfMerge.iterrows():
        if row['Layer'] == 'Conv2d':
            dfMerge.loc[index, 'MAC'] = row['OFmapX'] * \
                row['OFmapY']*row['ChO']*row['ChI']*row['Kernel']**2
            dfMerge.loc[index, 'ParamW'] = row['Kernel']**2 * \
                row['ChI'] * row['ChO']
            dfMerge.loc[index, 'ParamIF'] = row['ChI'] * \
                row['IFmapX'] * row['IFmapY']
            dfMerge.loc[index, 'ParamOF'] = row['ChO'] * \
                row['OFmapX'] * row['OFmapY']
            dfMerge.loc[index, 'MAC2W'] = dfMerge.loc[index,
                                                      'MAC'] / dfMerge.loc[index, 'ParamW']
            dfMerge.loc[index, 'MAC2IF'] = dfMerge.loc[index,
                                                       'MAC'] / dfMerge.loc[index, 'ParamIF']
            dfMerge.loc[index, 'MAC2OF'] = dfMerge.loc[index,
                                                       'MAC'] / dfMerge.loc[index, 'ParamOF']
        elif row['Layer'] == 'Linear':
            dfMerge.loc[index, 'MAC'] = row['ChO']*row['ChI']
            dfMerge.loc[index, 'ParamW'] = (
                row['ChI']+1) * row['ChO']  # add bias
            dfMerge.loc[index, 'ParamIF'] = row['ChI']
            dfMerge.loc[index, 'ParamOF'] = row['ChO']
            dfMerge.loc[index, 'MAC2W'] = dfMerge.loc[index,
                                                      'MAC'] / dfMerge.loc[index, 'ParamW']
            dfMerge.loc[index, 'MAC2IF'] = dfMerge.loc[index,
                                                       'MAC'] / dfMerge.loc[index, 'ParamIF']
            dfMerge.loc[index, 'MAC2OF'] = dfMerge.loc[index,
                                                       'MAC'] / dfMerge.loc[index, 'ParamOF']
        elif row['Layer'] == 'BatchNorm2d':
            dfMerge.loc[index, 'MAC'] = row['ChI']
            dfMerge.loc[index, 'ParamW'] = row['ChI'] * 2
        else:
            dfMerge.loc[index, 'MAC'] = 0
            dfMerge.loc[index, 'ParamW'] = 0

    # print(dfMerge)
    paramWSum = dfMerge['ParamW'].sum()
    logger.debug("Param match? : %s" %
                 (dfMerge['Param'].equals(dfMerge['ParamW'])))
    logger.debug("ParamW = %s" % (paramWSum))
    return dfMerge


def dfPlot(df):

    dfPlot = df[df['Layer'] == 'Conv2d']
    dfPlot = dfPlot[['Layer', 'ChI', 'IFmapX', 'IFmapY', 'ChO', 'OFmapX',
                     'OFmapY', 'ParamW', 'MAC', 'MAC2W', 'MAC2IF', 'MAC2OF']]
    dfPlot.reset_index(inplace=True, drop=True)

    logger.debug("dfPlot size = \n%s" % (dfPlot.count()))
    logger.debug("dfPlot = \n%s" % (dfPlot))

    plt.style.use("ggplot")
    plt.figure()
    #plt.plot(dfPlot.index, dfPlot['ParamW'])
    plt.plot(dfPlot.index, dfPlot['MAC2W'])
    plt.plot(dfPlot.index, dfPlot['MAC2IF'])
    plt.plot(dfPlot.index, dfPlot['MAC2OF'])
    plt.xlabel('layer')
    plt.ylabel('loss')
    plt.show()


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
    ap.add_argument('-v', '--verbose', action='store_true',
                    dest='verbose', help='Enabling debug info')
    args = vars(ap.parse_args())

    outfile = open(args["outfile"], 'w', encoding='UTF-8')

    if args['verbose']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    logger.debug("args = %s" % (args))

    #data = parse_file(args["infile"])
    parse_file_2(args["infile"])

    # print(data)

    # for row in df:
    #  print(row.rstrip())
    # for row in df:
    #  print(row.rstrip(), file=outfile)


if __name__ == "__main__":
    #debug = True
    # if debug:
    #  print(sys.argv)
    #  print(sys.argv[1:]) # ignore script name

    parse(sys.argv)
