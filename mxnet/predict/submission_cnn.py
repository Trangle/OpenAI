import pandas as pd
import os
import time as time

## Receives an array with probabilities for each class (columns) X images in test set (as listed in test.lst) and formats in Kaggle submission format, saves and compresses in submission_path

def gen_sub(predictions,test_lst_path="va.lst",submission_path="submission.csv"):

    ## append time to avoid overwriting previous submissions
    ## submission_path=time.strftime("%Y%m%d%H%M%S_")+submission_path

    ### Make submission
    ## check sampleSubmission.csv from kaggle website to view submission format
    header = "cat,cavy,chipmuck,dog,fox,giraffe,hyena,reindeer,spotted_deer,squirrel,wolf,yellow_weasel".split(',')

    # read first line to know the number of columns and column to use
    img_lst = pd.read_csv(test_lst_path,sep="/",header=None, nrows=1) 
    columns = img_lst.columns.tolist() # get the columns
    cols_to_use = columns[len(columns)-1] # drop the last one
    cols_to_use= map(int, str(cols_to_use)) ## convert scalar to list


    img_lst= pd.read_csv(test_lst_path,sep="/",header=None, usecols=cols_to_use) ## reads lst, use / as sep to goet last column with filenames

    img_lst=img_lst.values.T.tolist()

    df = pd.DataFrame(predictions,columns = header, index=img_lst)
    # f = lambda x: x.max()
    # x = df.apply(f, axis=1)
    # print(x)
    df.index.name = 'image'

    print("Saving csv to %s" % submission_path)
    df.to_csv(submission_path)
     
    print("Compress with gzip")
    os.system("gzip -f %s" % submission_path)
    
    print("stored in %s.gz" % submission_path)

   


