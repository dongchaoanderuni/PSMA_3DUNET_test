# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from train import config

def get_bone_lesion(data):
    output = data == 1
    output.dtype = np.uint8
    return output
def get_lymphNode_lesion(data):
    output = data == 2
    output.dtype = np.uint8
    return output
def get_prostate_lesion(data):
    output = data == 3
    output.dtype = np.uint8
    return output

def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

prediction_dir = os.path.abspath("../data/prediction")
def main():
    header_choose = ("boneLesion","lymphNodeLesion","localProstateLesion")
    masking_functions_choose = (get_bone_lesion, get_lymphNode_lesion, get_prostate_lesion)
    headerlist = []
    masking_functions_list = []
    for i in range(len(header_choose)):
        if (i+1) in config["labels"]:
            headerlist.append(header_choose[i])
            masking_functions_list.append(masking_functions_choose[i])
    header = tuple(headerlist)
    masking_functions = tuple(masking_functions_list)
    rows = list()
    for case_folder in glob.glob(os.path.join(prediction_dir,"validation_case*")):
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
    df = pd.DataFrame.from_records(rows, columns=header)
    df.to_csv(os.path.join(prediction_dir,"brats_scores.csv"))

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]
    
    plt.figure(1)
    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("../data/Restore/sample_1/validation_scores_boxplot.png")
    plt.show()
    plt.close()

    training_df = pd.read_csv("./training.log").set_index('epoch')

    plt.figure(2)
    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig('../data/Restore/sample_1/loss_graph.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
