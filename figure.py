
from utils.display import plot_accuracy_curves

# save_dir = '../figures/indian_pines1.png'
# save_dir = '../figures/indian_pines2.png'
save_dir = '../figures/indian_pines1_.png'
# save_dir = '../figures/indian_pines2_.png'
acc_dir = '../results1/accuracy.csv'
x_dict = {'data.train_prop': [0.05, 0.1, 0.15, 0.2]}
# x_dict = {'data.train_prop': [50, 100, 150, 200, 250, 300]}
y_field = 'test_oa'
legend_dict = {'method_name': ['3D-CNN-Hamida', '3D-CNN-Li', 'DeepFCN', 'SSRN',
                               'HybridSN', 'SAE', 'SVM', 'RandomForest']}

kwargs = {
    'fixed_field': {'data.data_name': 'indian_pines'},

    'title': 'Accuracy curves',
    'x_label': 'Training Proportion',
    'y_label': 'Test OA (%)',

    'title_font': {'family': 'sans-serif', 'weight': 'normal', 'size': 18},
    'label_font': {'family': 'Times New Roman', 'weight': 'normal', 'size': 15},
    'ticks_font': {'fontproperties': 'Times New Roman', 'size': 12},
    'legend_font': {'family': 'Times New Roman', 'weight': 'bold', 'size': 12},
    'legend_loc': 'lower right',
    'legend_frame': 0.8,
    'legend_ncol': 2,

    'line_colors': ['r', 'g', 'b', 'y', 'k'],
    'line_markers': ['o', '+', '*', '^'],
    'line_markersizes': 10,
    'line_styles': ['-', '-.', ':'],
    'line_fillstyles': 'none',

    'figsize': (8, 6),

    'save_dir': save_dir,
    'show': False
}

plot_accuracy_curves(acc_dir, x_dict, y_field, legend_dict, **kwargs)














