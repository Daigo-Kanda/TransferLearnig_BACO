import stochastic_transfer_eye_tracker
import glob
import os
import ITrackerData_person_tensor as ds
import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.__version__)
print(device_lib.list_local_devices())

# hyper parameter
dataset_path = "/kanda_tmp/GazeCapture_pre"
model_path = "model/models.046-2.46558.hdf5"
participants_num = 50
loop_num = 5
batch_size = "256"
image_size = "224"

participants_path = glob.glob(os.path.join(dataset_path, "**"))

participants_count = []
k = 0
for i, participant_path in enumerate(participants_path):
    metaFile = os.path.join(participant_path, 'metadata_person.mat')

    if os.path.exists(metaFile):
        participants_count.append(len(ds.loadMetadata(metaFile)['frameIndex']))
    else:
        participants_count.append(0)

tmp = zip(participants_count, participants_path)

# sorting
sorted_tmp = sorted(tmp, reverse=True)
participants_count, participants_path = zip(*sorted_tmp)

for i in reversed(range(participants_num)):
    for j in range(loop_num):
        # parser = sw_pathnetmod_tournament_eye_tracker.get_parser()
        # sw_pathnetmod_tournament_eye_tracker.main(parser.parse_args(
        #     [participants_path[i], "./my_stepwise/{}".format(participants_path[i][-5:]), "--image_size", image_size,
        #      "--batch_size", batch_size,
        #      "--epochs", "100", "--trained_model", model_path, "--transfer_all"])
        # )

        parser = stochastic_transfer_eye_tracker.get_parser()
        stochastic_transfer_eye_tracker.main(parser.parse_args(
            [participants_path[i], "./baco_transfer/{}".format(participants_path[i][-5:]), "--image_size",
             image_size, "--batch_size", batch_size,
             "--epochs", "100", "--trained_model", model_path, "--transfer_all", "--do_original"])
        )
