import json


path = '/mnt/sda/manli/HOI_KP/STIP-keypoints/v-coco/data/vcoco/vcoco_train.json'

with open(path, 'rt') as f:
    vcoco = json.load(f)

print(vcoco[0].keys())