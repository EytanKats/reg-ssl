import io
import os
import json
import glob
import random


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


def get_pairs(mr, ct):

    pairs = []
    for i in range(len(mr)):
        for j in range(len(ct)):

            name_mr = os.path.basename(mr[i])
            name_ct = os.path.basename(ct[j])
            entry = {
                'image_1': os.path.join(IMAGE_DIR, name_mr),
                'image_2': os.path.join(IMAGE_DIR, name_ct),
                'seg_1': os.path.join(MASK_DIR, name_mr),
                'seg_2': os.path.join(MASK_DIR, name_ct)
            }

            pairs.append(entry)

    return pairs


ROOT_DIR = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/abdomen_mrct/'
IMAGE_DIR = f'imagesTr'
MASK_DIR = f'labelsTr'

OUTPUT_FILE_PATH_TEMPLATE = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/abdomen_mrct/abdomen_mrct'

mr = glob.glob(os.path.join(ROOT_DIR, IMAGE_DIR, 'AbdomenMRCT_*_0000.nii.gz'))
ct = glob.glob(os.path.join(ROOT_DIR, IMAGE_DIR, 'AbdomenMRCT_*_0001.nii.gz'))

train_pairs = get_pairs(mr[15:], ct[15:])
val_pairs = get_pairs(mr[:15], ct[:15])
test_pairs = []

train_pairs = random.sample(train_pairs, 100)
val_pairs = random.sample(val_pairs, 20)

for i, train_pair in enumerate(train_pairs):
    train_pair['idx'] = i

output_file_path = OUTPUT_FILE_PATH_TEMPLATE + '_orig' + '.json'
create_json_data_file(
    output_file_path,
    train_pairs,
    val_pairs,
    test_pairs
)

