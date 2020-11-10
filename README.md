# CS_T0828_HW1
Code for *Selected Topics in Visual Recognition
using Deep Learning(2020 Autumn)* HW1

## Hardware
The following specs were used to create the original solution.
* ubuntu 16.04 LTS
* Intel(R) Core(TM) i9-10900 CPU @ 3.70GHz x 20
* 3x RTX 2080 Ti

## Reproducing Submission
To Reproduct the submission, do the folowed steps

1. [ Environment Setting](#Environment-Setting)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)
4. [Testing](#Testing)

## Environment Setting
My Environment settings are below:
* Python = 3.7.9
* pandas = 1.1.3
* torch = 1.7.0
* torchvision = 0.8.1

please check the version of packges before running the code especially torch and torchvision, or the code may not work.

## Dataset Preparation
Please download the dataset from https://www.kaggle.com/c/cs-t0828-2020-hw1/data
and unzip it. 
```
$ kaggle competitions download -c cs-t0828-2020-hw1
$ unzip cs-t0828-2020-hw1.zip
$ mkdir models 
$ ls
```
The whole working directory is structured as
```
working_dir
    +- models *your models generated at each epoch will be saved here*
    +- testing_data
    |    +- testing_data
    |    | 000004.jpg
    |    | ...
    +- training_data
    |    +-training_data
    |    | 000001.jpg
    |    |...
    training_labels.csv
    code.py
    label_id.csv
   
```
## Training
### Train models 
To train the model, run following commands.
`$ python3 code.py`

It will generate 2 types of file after training:
1. *EpochN.pkl* in directory models for N in 0~(epoch number-1), saves the model after each epoch. 
2. loss_acc.csv, saves the training loss of each epoch

If there is no file named *label_id* which saves the corresponding car label and id, it will also generate the file. The csv file should be like this:

| None                          | 0   |
| ----------------------------- | --- |
| BMW 3 Series Wagon 2012       | 0   |
| Ferrari 458 Italia Coupe 2012 | 1   |
|    ...                        | ... |
### Load old models and train for more epochs
To load existed model and continue training, run the commands followed:
```
$ python3 code.py loadmodel path_to_existed_model
```
***warning:***
It will overwrite the lossacc.csv file, so you have to rename the old file if want to keep the statistics

After training, it will generate models named as *Epoch+N.pkl* for N in 0~(epoch num-1)in the models directory.

## Testing
To test the model, run following commands:
`$ python3 code.py test path_to _model`

It will generate output.txt like this:
| id     | label                            |
| ------ | -------------------------------- |
| 001624 | Audi S6 Sedan 2011               |
| 013328 | Mercedes-Benz C-Class Sedan 2012 |
| ...    | ...                              |
