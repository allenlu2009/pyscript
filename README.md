# Collections of Python Script

## plot_parse.py
- test


## modelsum2csv.py and genmodel.py
* genmodel.py to generate model shape and layer description, redirect to a log file 
* modelsum2csv convert the pytorch summary(model()) or keras model(summary) to csv file.
* The csv file can be used for SCALE-sim to parse layer-by-layer performance.
* The script parse the model.log file and extract layer type (conv2d), shape, param