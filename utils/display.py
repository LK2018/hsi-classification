# -*- coding: utf-8 -*-

import pdb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.switch_backend('agg')

from .utils import check_path

__all__ = ['display_map_and_save', 'display_maps_and_save',
           'plot_loss_and_accuracy', 'plot_accuracy_curves']

map_colors = ['#FFFFFF', '#B0C4DE', '#E9967A', '#AFEEEE', '#BC8F8F', '#66CDAA',
              '#7B68EE', '#FF7F50', '#5F9EA0',
              '#3CB371', '#DA70D6', '#90EE90', '#4682B4', '#FAA460', '#9ACD32',
              '#6B8E23', '#778899']
markers = ['s', 'o', '^', 'd', 'P', '*', 'x', '+', '3', 'D']
line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'saddlebrown', 'orange',
               'yellow', 'slateblue']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (5, 5)),
              (0, (5, 1)), (0, (3, 5, 1, 5, 1, 5)),
              (0, (1, 1)), (0, (1, 5))]


def display_map_and_save(map, save_dir, show=False):
    """display a 2D array(np.ndarray), and save it as a '.png' file

    Parameters
    ----------
    map: ndarray: h*w
    save_dir: str, e.g. './classification_map.png'
    show: bool, if show map in the open window, default is False

    Returns
    -------
     no returns

    """

    save_dir = check_path(save_dir)

    cmap = colors.ListedColormap(map_colors, 'indexed')
    norm = colors.Normalize(vmin=0, vmax=16)
    height, width = map.shape[:2]
    fig, ax = plt.subplots()
    ax.imshow(map, aspect='equal', cmap=cmap, norm=norm)
    plt.axis('off')
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    fig.set_size_inches(width / 300, height / 300)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0, 0)

    plt.savefig(save_dir, dpi=300)
    if show:
        plt.show()
    plt.close()


def display_maps_and_save(image_data, save_dir, title=None, show=False):
    """display multi 2D data(np.ndarray) in a figure, and save as a '.png' file

    Parameters
    ----------
    image_data: dict,
    e.g. {'raw image': **, 'ground truth': **, 'classification map': **, ...}
    save_dir: str, e.g. './maps.png'
    title: str, e.g. 'Training results display'
    show: bool, if show map in the open window, default is False

    Returns
    -------
    no returns

    """

    save_dir = check_path(save_dir)

    # sub_plot_num = len(image_data) + 1
    # row, col = 2, np.ceil(sub_plot_num/2.).astype(np.int)
    sub_plot_num = len(image_data)
    row, col = np.ceil(sub_plot_num / 4.).astype(np.int), min(4, sub_plot_num)

    plt.figure(figsize=(12, 8))
    grid = plt.GridSpec(row, col, wspace=0.25, hspace=0.25)
    cmap = colors.ListedColormap(map_colors, 'indexed')
    norm = colors.Normalize(vmin=0, vmax=16)

    it = iter(image_data.items())
    for i in range(row):
        for j in range(col):
            try:
                key, value = next(it)
            except StopIteration:
                break  # break the inner loop, can work
            ax = plt.subplot(grid[i, j])
            ax.imshow(value, aspect='equal', cmap=cmap, norm=norm)
            ax.axis('off')
            ax.set_title(key, fontsize=12)

    if title is not None:
        plt.suptitle(title, fontsize=14)
    plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


def plot_loss_and_accuracy(graph_data, save_dir, title=None, show=False):
    """plot multi 1D array(np.ndarray) such as loss, train_accuracy etc. in a
    figure and save as a '.png' file

    Parameters
    ----------
    graph_data: dict,
    e.g. {'loss': **, 'train accuracy': **, 'test accuracy': **}
    graph_data dictionary only contains loss list/ndarray and accuracy(train,
    val, test, ...) list/ndarray
    save_dir: str, the output graph save dir
    title: graph's title, default is None
    show: bool, if show map in the open window, default is False

    Returns
    -------
    no returns

    """

    save_dir = check_path(save_dir)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax_ = ax.twinx()

    i = 0
    lines = []
    for key, value in graph_data.items():
        if key == 'loss':
            l, = ax_.plot(range(1, len(value) + 1), value, color=line_colors[i],
                          label=key, linewidth=0.8)
            ax_.set_ylabel('loss', fontsize=10)
            ax_.tick_params(labelsize=8)
        else:
            l, = ax.plot(range(1, len(value) + 1), [v * 100 for v in value],
                         color=line_colors[i],
                         label=key, linewidth=0.8)
        lines.append(l)
        i += 1
    ax.set_ylabel('accuracy(%)', fontsize=10)
    ax.set_xlabel('epoch', fontsize=10)
    ax.tick_params(labelsize=8)
    legend_label = [l.get_label() for l in lines]
    ax_.legend(lines, legend_label, loc='best', fontsize=8, framealpha=0.8)

    if title is not None:
        ax.set_title(title, fontsize=13)
    plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


# def plot_accuracy_curves(acc_df, save_dir, x_variable='train_prop',
#                          y_variable='test_accuracy',
#                          xlabel='Train Proportion (%)', ylabel='Test OA (%)',
#                          title=None,
#                          compared_variable='method',
#                          fixed_variable=None, plot_params=None,
#                          legend_params=None, show=False):
#     """plot curves to see the relationship between the variables, for example,
#     training prop and test overall accuracy, variables data are extracted from
#     pd.DataFrame data, figure will be saved as a '.png' file
#
#     Parameters
#     ----------
#     acc_df: pd.DataFrame, accuracy data, like:
#     ===========================================================================
#     |   method    |  train_prop | train_accuracy | val_accuracy | test_accuracy
#     +-------------+-------------+----------------+--------------+--------------
#     |    m1       |      0.05   |       0.99     |      0.90    |      0.91
#     +-------------+-------------+----------------+--------------+--------------
#     |    m2       |      0.05   |       0.98     |      0.93    |      0.92
#     +-------------+-------------+----------------+--------------+--------------
#     |    ...
#     +-------------------
#     save_dir: str, graph file save path, e.g. './accuracy.png'
#     x_variable: str, variable of axis x, must be contained in the acc_df's
#     column names, e.g. 'train_prop'
#     y_variable: str, variable of axis y, must be contained in the acc_df's
#     column names, e.g. 'test_accuracy'
#     xlabel：str, label of  axis x, e.g. 'Train Proportion(%)'
#     ylabel: str, label of axis y, e.g. 'Test OA(%)'
#     compared_variable: str, variable will be compared, such as 'method'
#     fixed_variable: dict, fix some variable's value
#     plot_params: dict, the params plot lines,
#     e.g. {'linewidth': 1.2, 'markersize': 8, 'fillstyle': 'none'}
#     legend_params: dict, the legend params,
#     e.g. {'loc': 'lower right', 'fontsize': 12, 'framealpha': 0.8}
#     show: bool, if show map in the open window, default is False
#
#     Returns
#     -------
#     no returns
#
#     """
#
#     save_dir = check_path(save_dir)
#
#     if fixed_variable is not None:
#         for key, value in fixed_variable.items():
#             acc_df = acc_df.loc[acc_df[key] == value]
#
#     x = np.unique(acc_df[x_variable].values)
#     # compared = np.unique(acc_df[compared_variable].values)
#     compared = ['DFRes(SS)', 'DFRes(Spa)', 'DFRes(Spe)', 'SSCNN', 'SSRN', 'SVM',
#                 '3D-CNN-Hamida', '3D-CNN-Li', 'DeepFCN']
#     compared = ['DFRes-CR', 'SSRN', 'SGL', 'SSCNN', 'Semi-2dCNN', 'SVM']
#     new_df = pd.DataFrame(index=compared, columns=x)
#     for idx in compared:
#         tem_df1 = acc_df.loc[acc_df[compared_variable] == idx]
#         for col in x:
#             tem_df2 = tem_df1.loc[tem_df1[x_variable] == col]
#             if len(tem_df2) > 1:
#                 print(
#                     "Warning! Plot accuracy curves, more than one row of data "
#                     "is selected, the first row is selected by default, you "
#                     "can set 'fixed_variable'."
#                     "\nPlease check it: "
#                     "\n{}".format(tem_df2))
#             new_df.loc[idx, col] = tem_df2[y_variable].values[0]
#
#     font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
#     tick_font = {'fontproperties': 'Times New Roman', 'size': 20}
#
#     plot_params_ = {'color': line_colors,
#                     'marker': markers,
#                     'linestyle': linestyles,
#                     'linewidth': 1.2,
#                     'markersize': 8,
#                     'fillstyle': 'none'}
#
#     legend_params_ = {'loc': 'lower right',
#                       'framealpha': 0.8,
#                       'prop': font}
#
#     if plot_params is not None:
#         plot_params = {k: v for d in [plot_params_, plot_params] for k, v in
#                        d.items()}
#     else:
#         plot_params = plot_params_
#     if legend_params is not None:
#         legend_params = {k: v for d in [legend_params_, legend_params] for k, v
#                          in d.items()}
#     else:
#         legend_params = legend_params_
#
#     assert len(new_df) <= len(markers)
#     f = plt.figure(figsize=(8, 6))
#     ax = f.gca()
#     ax.set_xticks(range(0, len(x)))
#     ax.set_xticklabels([str(i) for i in x])
#     plt.xticks(**tick_font)
#     plt.yticks(**tick_font)
#
#     if title is not None:
#         plt.title(title, fontdict=font)
#
#     # ax.set_ylim(82, 95)
#
#     lines = []
#     plot_params_ = plot_params.copy()
#     for i, y in enumerate(new_df.values):
#         plot_params_['marker'] = plot_params['marker'][i]
#         plot_params_['color'] = plot_params['color'][i]
#         plot_params_['linestyle'] = plot_params['linestyle'][i]
#         l, = ax.plot(y * 100, **plot_params_)
#         lines.append(l)
#     ax.set_xlabel(xlabel, fontdict=font)
#     ax.set_ylabel(ylabel, fontdict=font)
#     # ax.grid(linestyle='--')  # not add grid
#     ax.legend(handles=lines, labels=list(compared), **legend_params)
#     plt.tight_layout()
#
#     plt.savefig(save_dir)
#     if show:
#         plt.show()
#     plt.close()


def plot_accuracy_curves(acc_dir, x_dict, y_field, legend_dict, **kwargs):

    """plot accuracy curves

    Parameters
    ----------
    acc_dir: accuracy dir, e.g. './accuracy.csv', a table like:

        ========================================================================
        |  method  |  train_prop | train_accuracy | val_accuracy | test_accuracy
        +----------+-------------+----------------+--------------+--------------
        |   m1     |      0.05   |       0.99     |      0.90    |      0.91
        +----------+-------------+----------------+--------------+--------------
        |   m2     |      0.05   |       0.98     |      0.93    |      0.92
        +----------+-------------+----------------+--------------+--------------
        |  ...
        +-------------------

    x_dict: x variable, dict, e.g. {'train_prop': [0.05, 0.1, 0.15, 0.2]}
    y_field: y variable, str, e.g. 'test_oa'
    legend_dict: compared variable, dict, e.g. {'method': [CNN, SSRN, SVM, ...]}
    kwargs: can pass other optional args, such as:

        fixed_field={'data_name': 'indian_pines', 'epoch': 2000, ...}

        title='Accuracy curves'
        x_lable='Training Proportion (%)'
        y_label='Test OA (%)'
        legend_label = ['CNN', 'SSRN', 'SVM', ...]

        title_font={'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        label_font={'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        ticks_font={'fontproperties': 'Times New Roman', 'size': 20}
        legend_font={'family': 'Times New Roman', 'weight': 'normal',
                     'size': 20}
        legend_loc='lower right' ('loc', legend location)
        legend_frame=0.8 ('framealpha', transparency)
        legend_ncol=1 ('ncol', number of columns of legend)

        line_colors=['r', 'g', 'b', 'y', 'k']
        line_markers=['o', '+', '*', '^']
        line_markersizes=1.2
        line_styles=['-', '-.', '.']
        line_fillstyles='none'

        figsize=(8, 6)

        save_dir='./figure.png'
        show=False (if open a widow to show the figure)

    Returns
    -------

    """

    acc_df = pd.read_csv(acc_dir)

    fixed_field = kwargs.get('fixed_field', None)
    if fixed_field is not None:
        for key, value in fixed_field.items():
            acc_df = acc_df.loc[acc_df[key] == value]

    x_field = list(x_dict.keys())[0]
    x_value = list(x_dict.values())[0]
    x_label = kwargs.get('x_label', x_field)
    y_label = kwargs.get('y_label', y_field)

    legend_field = list(legend_dict.keys())[0]
    legend_value = list(legend_dict.values())[0]
    legend_label = kwargs.get('legend_label', legend_value)

    new_df = pd.DataFrame(index=legend_value, columns=x_value)
    for idx in legend_value:
        tem_df1 = acc_df.loc[acc_df[legend_field] == idx]
        for col in x_value:
            tem_df2 = tem_df1.loc[tem_df1[x_field] == col]
            if len(tem_df2) > 1:
                print(
                    "Warning! Plot accuracy curves, more than one row of data "
                    "is selected, the first row is selected by default, you "
                    "can set 'fixed_variable'."
                    "\nPlease check it: "
                    "\n{}".format(tem_df2))
            new_df.loc[idx, col] = tem_df2[y_field].values[0]

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    title_font = kwargs.get('title_font', font)
    label_font = kwargs.get('label_font', font)
    legend_font = kwargs.get('legend_font', font)
    ticks_font = kwargs.get('ticks_font', {'fontproperties': 'Times New Roman',
                                           'size': 20})

    color_v = kwargs.get('line_colors', line_colors)
    marker_v = kwargs.get('lines_markers', markers)
    markersize_v = kwargs.get('line_markersizes', 1.2)
    linestyle_v = kwargs.get('line_styles', linestyles)
    fillstyle_v = kwargs.get('line_fillstyles', 'none')

    plot_params = []
    plot_params_str = filter(lambda s: s.split('_')[1] == 'v',
                             list(locals().keys()))
    plot_params = []
    plot_params_str = list(filter(lambda s: s[-2:] == '_v',
                                  list(locals().keys())))
    ii = np.zeros(len(plot_params_str)).astype(int)
    while len(plot_params) < len(new_df):
        plot_params_i = {}
        for j, variable_str in enumerate(plot_params_str):
            variable_value = locals()[variable_str]
            variable_str_new = variable_str.split('_')[0]
            if isinstance(variable_value, list):
                ii[j] = ii[j] if ii[j] < len(variable_value) else 0
                plot_params_i[variable_str_new] = variable_value[ii[j]]
            else:
                plot_params_i[variable_str_new] = variable_value
        plot_params.append(plot_params_i)
        ii += 1

    legend_params = {'loc': kwargs.get('legend_loc', 'lower right'),
                     'framealpha': kwargs.get('legend_frame', 0.8),
                     'ncol': kwargs.get('legend_ncol', 1),
                     'prop': legend_font}

    figsize=kwargs.get('figsize', (8, 6))
    f = plt.figure(figsize=figsize)
    ax = f.gca()
    ax.set_xticks(range(0, len(x_value)))
    ax.set_xticklabels([str(i) for i in x_value])
    plt.xticks(**ticks_font)
    plt.yticks(**ticks_font)

    title = kwargs.get('title', None)
    if title is not None:
        plt.title(title, fontdict=title_font)

    lines = []
    plot_params_ = plot_params.copy()
    for i, y in enumerate(new_df.values):
        l, = ax.plot(y * 100, **plot_params[i])
        lines.append(l)
    ax.set_xlabel(x_label, fontdict=label_font)
    ax.set_ylabel(y_label, fontdict=label_font)
    ax.legend(handles=lines, labels=legend_label, **legend_params)
    plt.tight_layout()

    save_dir = kwargs.get('save_dir', './figure.png')
    show = kwargs.get('show', False)
    plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # test plot_accuracy_curves()
    # df = pd.read_csv('./all_result.csv')
    # plot_accuracy_curves(df,
    #                      './cc.png',
    #                      x_variable='var',
    #                      y_variable='Best_test_acc',
    #                      compared_variable='NET_TYPE',
    #                      xlabel='var',
    #                      ylabel='test oa',
    #                      fixed_variable={'NET_TYPE': 'ResNet', 'prop': 0.2}
    #                      )
    # plot_accuracy_curves(df,
    #                      './bb.png',
    #                      x_variable='prop',
    #                      y_variable='Best_test_acc',
    #                      compared_variable='NET_TYPE',
    #                      xlabel='prop',
    #                      ylabel='test oa',
    #                      fixed_variable={'var': 0}
    #                      )
    #
    # # pred 145*145  pred.png
    # pred = np.zeros((145, 145))
    # gt = np.zeros((145, 145))
    # pred_ = pred.copy()
    # pred_[gt == 0] = 0
    # # pred_.png pred_.mat
    # # import scipy.io as sio
    # # pred_ = sio.loadmat('./pred_.mat')['pred_']
    # # sio.savemat('./pred_.mat', {'pred_': pred_}
    # # display_map_and_save(pred_, './pred_.png')

    # test display_map_and_save()
    import scipy.io as sio

    # gt = 'c:\\Users\\likui\\Desktop\\HSIClassification\\hsi_data\\' \
    #      'salinas\\Salinas_gt.mat'
    # gt = sio.loadmat(gt)['salinas_gt']
    # gt = 'c:\\Users\\likui\\Desktop\\HSIClassification\\hsi_data\\' \
    #      'indian_pines\\Indian_pines_gt.mat'
    # gt = sio.loadmat(gt)['indian_pines_gt']
    gt = 'c:\\Users\\likui\\Desktop\\HSIClassification\\hsi_data\\' \
         'paviau\\PaviaU_gt.mat'
    gt = sio.loadmat(gt)['paviaU_gt']
    print(gt.shape)
    display_map_and_save(gt, './gt.png')

