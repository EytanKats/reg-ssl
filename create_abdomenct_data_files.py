import io
import os
import json


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


def get_pairs(indices):

    pairs = []
    for i in range(len(indices)):
        for j in range(len(indices)):

            if i >= j:
                continue

            nu1 = indices[i]
            nu2 = indices[j]
            entry = {
                'image_1': os.path.join(IMAGE_DIR, f'AbdomenCTCT_{str(nu1).zfill(4)}_0000.nii.gz'),
                'image_2': os.path.join(IMAGE_DIR, f'AbdomenCTCT_{str(nu2).zfill(4)}_0000.nii.gz'),
                'seg_1': os.path.join(MASK_DIR, f'AbdomenCTCT_{str(nu1).zfill(4)}_0000.nii.gz'),
                'seg_2': os.path.join(MASK_DIR, f'AbdomenCTCT_{str(nu2).zfill(4)}_0000.nii.gz')
            }

            pairs.append(entry)

    return pairs


ROOT_DIR = f'/home/xxxx/storage/staff/yyyyyxxxx/projects/reg_ssl/data/abdomen_ctct'
IMAGE_DIR = f'imagesTr'
MASK_DIR = f'labelsTr'

OUTPUT_FILE_PATH_TEMPLATE = f'/home/xxxx/storage/staff/yyyyyxxxx/projects/reg_ssl/data/abdomen_ctct/abdomen_ct'

train_idx = (2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30)
val_idx = (2, 8, 14, 20, 26, 30)
test_idx = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28)

train_pairs = get_pairs(train_idx)
val_pairs = get_pairs(val_idx)
test_pairs = get_pairs(test_idx)

for i, train_pair in enumerate(train_pairs):
    train_pair['idx'] = i

output_file_path = OUTPUT_FILE_PATH_TEMPLATE + '_orig' + '.json'
create_json_data_file(
    output_file_path,
    train_pairs,
    val_pairs,
    test_pairs
)

