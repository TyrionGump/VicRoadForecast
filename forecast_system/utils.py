# -*- coding: utf-8 -*-
"""
@File:utils.py
@Date: 2021/12/3
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import os
import numpy as np
from config import args


def save_raw_data(data: dict, root_path: str):
    root_path = root_path.replace(':', '-')
    os.makedirs(root_path, exist_ok=True)
    for link_id in data.keys():
        data[link_id].to_csv(root_path + '/{}.csv'.format(link_id), index=False)


def save_dataset(training_set: dict, testing_set: dict, root_path):
    dataset_num = 1
    while os.path.exists(root_path + '/{}'.format(dataset_num)):
        dataset_num += 1

    training_set_path = root_path + '/{}/training_set'.format(dataset_num)
    testing_set_path = root_path + '/{}/testing_set'.format(dataset_num)

    os.makedirs(training_set_path + '/feature/', exist_ok=True)
    os.makedirs(training_set_path + '/target/', exist_ok=True)
    os.makedirs(testing_set_path + '/feature/', exist_ok=True)
    os.makedirs(testing_set_path + '/target/', exist_ok=True)

    link_ids = training_set['feature'].keys()

    for link_id in link_ids:
        np.save(training_set_path + '/feature/{}.npy'.format(link_id), training_set['feature'][link_id])
        np.save(training_set_path + '/target/{}.npy'.format(link_id), training_set['target'][link_id])
        np.save(testing_set_path + '/feature/{}.npy'.format(link_id), testing_set['feature'][link_id])
        np.save(testing_set_path + '/target/{}.npy'.format(link_id), testing_set['target'][link_id])

    with open(root_path + '/{}/dataset_info.txt'.format(dataset_num), 'w') as f:
        f.write('start training time: ' + str(args.research_config['period']['TRAINING_START_TIME']) + '\n')
        f.write('end training time: ' + str(args.research_config['period']['TRAINING_END_TIME']) + '\n')
        f.write('start testing time: ' + str(args.research_config['period']['TESTING_START_TIME']) + '\n')
        f.write('end testing time: ' + str(args.research_config['period']['TESTING_END_TIME']) + '\n')
        f.write('forward steps: ' + str(args.research_config['window']['FORWARD_STEPS']) + '\n')
        f.write('backward steps: ' + str(args.research_config['window']['BACKWARD_STEPS']) + '\n')
        f.write('neighbours_feature: ' + str(args.neighbours_feature) + '\n')
        f.write('temporal_feature: ' + str(args.temporal_feature) + '\n')
        f.write('link_feature: ' + str(args.link_feature) + '\n')
        f.write('training feature shape for each link: ' + str(training_set['feature'][link_id].shape)+ '\n')
        f.write('training target shape for each link: ' + str(training_set['target'][link_id].shape) + '\n')
        f.write('testing feature shape for each link: ' + str(testing_set['feature'][link_id].shape) + '\n')
        f.write('testing target shape for each link: ' + str(testing_set['target'][link_id].shape) + '\n')



