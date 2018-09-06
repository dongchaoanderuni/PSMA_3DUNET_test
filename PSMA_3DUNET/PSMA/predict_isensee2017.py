import os
import subprocess

from train_isensee2017 import config
from keras_contrib.layers.normalization import InstanceNormalization
from unet3d.prediction import run_validation_case
from unet3d.utils import pickle_load


def main():
    prediction_dir = os.path.abspath("../data/prediction_isensee2017")
    if not os.path.exists(prediction_dir):
        subprocess.call('mkdir' + '-p' + prediction_dir, shell=True)

    validation_indices = pickle_load(config["validation_file"])
    for i in range(len(validation_indices)):
        run_validation_case(test_index=i, out_dir=os.path.join(prediction_dir, "validation_case_{}".format(i)),
                            model_file=config["model_file"], validation_keys_file=config["validation_file"],
                            training_modalities=config["training_modalities"], output_label_map=True,
                            labels=config["labels"], hdf5_file=config["data_file"])


if __name__ == "__main__":
    main()
