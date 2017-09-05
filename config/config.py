from config_rand1 import *

# there are some experiment settings for different config file
# basic params are same among all files, so the only differences will appear in train params or data params
# net params are not in the config file,maybe should, such as hidden layer dimensions of lstm
# config1 and config2 are both for test set 0,4,7 and training set 1,2,3,6,so the only difference is the loss function
# change would be made for the attr_num, maybe 12 better
# config1 and config3 are only different in attr_num ,8 or 12
# other will help choose the test data to show a more pretty result
