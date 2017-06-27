# neural_image_captioning
Neural image captioning implementation with keras (tensorflow/theano).

Instructions to train from zero using the iapr2012 dataset (or any other dataset)

download the file from these website
http://imageclef.org/photodata

move the downloaded file to the datasets/IAPR_2012/ directory

untar the file
tar xvf iaprtc12.tgz

edit the file train.py by changing the flag extract_image_features to True.

run the train script 
python3 train.py

Instructions to test data (observe results from a model trained on iapr2012 dataset)


