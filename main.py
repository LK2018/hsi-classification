import importlib

from config import get_params2, method_name_dict


method_names = [
                '3D-CNN-Li',
                '3D-CNN-Hamida',
                'DeepFCN',
                'SSRN',
                'HybridSN',
                'SAE',
                # 'DBN',
                'SVM',
                'RandomForest'
]
data_names = [
              'indian_pines',
              'pavia_university',
              'salinas',
              'ksc'
]
train_props = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.20, 50., 100., 150., 200.,
               250., 300.]
val_props = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 100., 100., 100., 100.,
             100., 100.]

override_params = None

for data_name in data_names:
    for method_name in method_names:
        for train_prop, val_prop in zip(train_props, val_props):

            override_params = [
                'gpu', 3,

                'data.train_prop', train_prop,
                'data.val_prop', val_prop,

                'result_dir', '../results',
                'folder_level', "['data_name', 'method_name', 'train_prop']"
            ]

            params = get_params2(method_name, data_name, override_params)
            method = importlib.import_module('methods.%s' %
                                             method_name_dict[method_name])
            method.run(params)

