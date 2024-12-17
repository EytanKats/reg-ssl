import io
import os
import json
import torch


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


ROOT_DIR = f'/home/xxxx/storage/staff/eytanxxxx/projects/reg_ssl/data/radchest_ct/'
IMAGE_DIR = f'imgs'
MASK_DIR = f'lbls_ts'
PAIRS = f'/home/xxxx/storage/staff/eytanxxxx/projects/reg_ssl/data/radchest_ct/list_of_pairs.pth'

OUTPUT_FILE_PATH_TEMPLATE = f'/home/xxxx/storage/staff/eytanxxxx/projects/reg_ssl/data/radchest_ct/radchest_ct'

NUM_SPLITS = 8
NUM_VAL_IMAGES = 20

# Get list of pairs
pairs = torch.load(os.path.join(PAIRS))

train_pairs, val_pairs, test_pairs = [], [], []
for pair in pairs:

    entry = {
        'image_1': os.path.join(IMAGE_DIR, f'{pair[0]}.nii.gz'),
        'image_2': os.path.join(IMAGE_DIR, f'{pair[1]}.nii.gz'),
        'seg_1': os.path.join(MASK_DIR, f'{pair[0]}.nii.gz'),
        'seg_2': os.path.join(MASK_DIR, f'{pair[1]}.nii.gz')
    }

    if pair[0].startswith('trn'):
        train_pairs.append(entry)
    elif pair[0].startswith('tst'):
        val_pairs.append(entry)
    elif pair[0].startswith('val'):
        test_pairs.append(entry)

for i, train_pair in enumerate(train_pairs):
    train_pair['idx'] = i

output_file_path = OUTPUT_FILE_PATH_TEMPLATE + '_orig' + '.json'
create_json_data_file(
    output_file_path,
    train_pairs,
    val_pairs,
    test_pairs
)

