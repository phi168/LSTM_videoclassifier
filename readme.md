WIP 
===============  

Usage:  
===============  
1. index videos:  
    > from data import DataSet  
    > d = DataSet( read_index = False)  
    > d.setup_data()  
2. train NN:  
    > from train import train  
    > train()  
3. categorise video:  
    > from test_if_fake import test_if_fake  
    > test_if_fake(path_to_video)  


directory structure:  
===============  
PreProcess.py  
train.py  
data.py  
test_if_fake.py  
/videos/  
/videos/fake/  
/videos/real/  


Files:  
===============  
PreProcess.py:  
---------------
Intended to help NN focus on the face and reduce file-size. Current solution is shaky and not implemented in the actual training.  

data.py:  
---------------
Contains DataSet class, which chops videos into chunks and parses them along for training.  

1. Do this only once:  
To preprocess data run  
> d = DataSet(read_index = False)  
> d.setup_data()  
Directory of files is kept in data_file.csv  

2. If data was already chopped up and indexed  
> d = DataSet()  

3. Get data for training:  
> x,y = d.get_data('train')  

train.py: 
---------------
Trains NN. Assumes data set was already setup using d.setup_data
> from train import train
> train()

test_if_fake.py:  
---------------
loads NN which was produce by train() and applies to input video  
test_if_fake(path_to_video) produces a processed video annotating which chunks it thinks are fake or real.  

