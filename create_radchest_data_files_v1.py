import io
import os
import json
import torch

from sklearn.model_selection import train_test_split, KFold


def create_json_data_file(
    output_file_path,
    training=None,
    validation=None,
    test=None
):

    # Create dictionary
    files_in_decathlon_format = {}
    if training is not None:
        files_in_decathlon_format['training'] = training
    if validation is not None:
        files_in_decathlon_format['validation'] = validation
    if test is not None:
        files_in_decathlon_format['test'] = test

    # Write json file
    with io.open(output_file_path, 'w', encoding='utf8') as output_file:
        json.dump(files_in_decathlon_format, output_file, indent=4, ensure_ascii=False)


ROOT_DIR = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/radchest_ct/'
IMAGE_DIR = f'imgs'
MASK_DIR = f'lbls_ts'
PAIRS = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/radchest_ct/list_of_pairs.pth'

OUTPUT_FILE_PATH_TEMPLATE = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/radchest_ct/radchest_ct'

NUM_SPLITS = 8
NUM_VAL_IMAGES = 20

# Get list of pairs
pairs = torch.load(os.path.join(PAIRS))
pairs = [{'image_1': os.path.join(IMAGE_DIR, f'{pairs[i][0]}.nii.gz'),
          'image_2': os.path.join(IMAGE_DIR, f'{pairs[i][1]}.nii.gz'),
          'seg_1': os.path.join(MASK_DIR, f'{pairs[i][0]}.nii.gz'),
          'seg_2': os.path.join(MASK_DIR, f'{pairs[i][1]}.nii.gz')}
         for i in range(len(pairs))]

# Split files to training and validation
kf = KFold(n_splits=NUM_SPLITS, random_state=1987, shuffle=True)
for fold, (train_val_idx, test_idx) in enumerate(kf.split(pairs)):

    train_val_idx = train_val_idx.tolist()
    train_val_pairs = [pairs[i] for i in train_val_idx]
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=NUM_VAL_IMAGES)

    test_idx = test_idx.tolist()
    test_pairs = [pairs[i] for i in test_idx]

    for i, train_pair in enumerate(train_pairs):
        train_pair['idx'] = i

    output_file_path = OUTPUT_FILE_PATH_TEMPLATE + '_fold' + str(fold) + '.json'
    create_json_data_file(
        output_file_path,
        train_pairs,
        val_pairs,
        test_pairs
    )

