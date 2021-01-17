import os
import scipy.io as sio
# from dataset import get_proportional_masks, get_fixed_number_masks


# indian_gt_ = '../data/indian_pines/Indian_pines_gt.mat'
# pavia_gt_ = '../data/paviau/PaviaU_gt.mat'
# ksc_gt_ = '../data/ksc/KSC_gt.mat'
# salinas_gt_ = '../data/salinas/Salinas_gt'
#
# indian_gt = sio.loadmat(indian_gt_)['indian_pines_gt']
# pavia_gt = sio.loadmat(pavia_gt_)['paviaU_gt']
# ksc_gt = sio.loadmat(ksc_gt_)['KSC_gt']
# salinas_gt = sio.loadmat(salinas_gt_)['salinas_gt']
#
# indian_save = '../data/indian_pines'
# paviau_save = '../data/paviau'
# ksc_save = '../data/ksc'
# salinas_save = '../data/salinas'
#
# train_props1 = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
# val_props1 = [0.05, 0.2]
# train_props2 = [50, 100, 150, 200, 250, 300]
# val_props2 = [100, 200]
#
# for train_prop in train_props1:
#     for val_prop in val_props1:
#         get_proportional_masks(indian_gt, train_prop, val_prop,
#                                save_dir=indian_save)
#         get_proportional_masks(pavia_gt, train_prop, val_prop,
#                                save_dir=paviau_save)
#         get_proportional_masks(ksc_gt, train_prop, val_prop,
#                                save_dir=ksc_save)
#         get_proportional_masks(salinas_gt, train_prop, val_prop,
#                                save_dir=salinas_save)
#
# for train_prop in train_props2:
#     for val_prop in val_props2:
#         get_fixed_number_masks(indian_gt, train_prop, val_prop,
#                                save_dir=indian_save)
#         get_fixed_number_masks(pavia_gt, train_prop, val_prop,
#                                save_dir=paviau_save)
#         get_fixed_number_masks(ksc_gt, train_prop, val_prop,
#                                save_dir=ksc_save)
#         get_fixed_number_masks(salinas_gt, train_prop, val_prop,
#                                save_dir=salinas_save)

#############################################################################

# tr = '../data/indian_pines/masks_50_100/train_mask.mat'
# va = '../data/indian_pines/masks_50_100/val_mask.mat'
# te = '../data/indian_pines/masks_50_100/test_mask.mat'
#
# tr = sio.loadmat(tr)['train_mask']
# va = sio.loadmat(va)['val_mask']
# te = sio.loadmat(te)['test_mask']
#
# print(len(tr[tr == 1]))
# print(len(va[va == 1]))
# print(len(te[te == 1]))
# print(len(tr[tr == 1]) + len(va[va == 1]) + len(te[te == 1]))
#
# import matplotlib.pyplot as plt
#
# plt.imshow(tr)
# plt.show()
# plt.imshow(va)
# plt.show()
# plt.imshow(te)
# plt.show()
# plt.imshow(tr + va + te)
# plt.show()

###################################### xiaochuan #############################

import numpy as np


def get_proportional_masks(target, train_prop, val_prop, num, save_dir=None):

    assert train_prop + val_prop < 1
    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        # at least 3 samples for training
        train_num = max(int(round(len(idx) * train_prop)), 3)
        val_num = max(int(round(len(idx) * val_prop)), 3)

        np.random.shuffle(idx)
        train_idx = idx[:train_num]
        val_idx = idx[train_num:train_num + val_num]
        test_idx = idx[train_num + val_num:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        folder_name = 'masks_{}_{}_{}'.format(train_prop, val_prop, num)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'),
                    {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'),
                    {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'),
                    {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


indian_gt_ = '../xiaochuan_data/indian_pines/Indian_pines_gt.mat'
pavia_gt_ = '../xiaochuan_data/paviau/PaviaU_gt.mat'

indian_gt = sio.loadmat(indian_gt_)['indian_pines_gt']
pavia_gt = sio.loadmat(pavia_gt_)['paviaU_gt']

indian_save = '../xiaochuan_data/indian_pines'
paviau_save = '../xiaochuan_data/paviau'

train_props1 = [0.1]
val_props1 = [0.05]

for i in range(1, 11):
    for train_prop in train_props1:
        for val_prop in val_props1:
            get_proportional_masks(indian_gt, train_prop, val_prop, i,
                                   save_dir=indian_save)
            get_proportional_masks(pavia_gt, train_prop, val_prop, i,
                                   save_dir=paviau_save)
