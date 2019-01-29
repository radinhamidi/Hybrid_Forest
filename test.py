import numpy as np
import time
from operator import itemgetter
import pandas as pd
from VFDT.VFDT import *
from VFDT.utils import *
import collections
from scipy import stats

plot = 1
report = 1


''' waveform dataset setting'''
test_numbers = 20
windows_size_array = [5, 15, 20, 30, 100]
hybrid_power_threshold_array = [0.2, 0.5, 0.6, 0.8, 1]
prob_threshold = 0.99
datasplit = 0.6

figsave_directory = './figures'
dataset_fig_dir = '/waveform'
datadir = './dataset'
dataset = '/waveform.data'

results_acc = np.zeros((3, len(windows_size_array), len(hybrid_power_threshold_array)))
results_std = np.zeros((3, len(windows_size_array), len(hybrid_power_threshold_array)))

data = pd.read_csv(datadir + dataset, low_memory=False)
n_samples = data.shape[0]

n_training = int(datasplit * n_samples)
train = data[:n_training]
test = data[n_training:]
title = list(data.columns.values)
features = title[:-1]
n_features = features.__len__()
del data
test = test[:1000]

''' madelon dataset setting
test_numbers = 3
windows_size_array = [5, 15, 20, 30, 100]
hybrid_power_threshold_array = [0.2, 0.5, 0.6, 0.8, 1]
prob_threshold = 0.99

figsave_directory = './figures'
dataset_fig_dir = '/madelon'
datadir = './dataset'
dataset_train = '/madelon_train.data'
dataset_train_labels = '/madelon_train.labels'
dataset_test = '/madelon_valid.data'
dataset_test_labels = '/madelon_valid.labels'


results_acc = np.zeros((3, len(windows_size_array), len(hybrid_power_threshold_array)))
results_std = np.zeros((3, len(windows_size_array), len(hybrid_power_threshold_array)))

data_train = pd.read_csv(datadir + dataset_train, low_memory=False, delimiter=' ', header=None)
data_train_labels = pd.read_csv(datadir + dataset_train_labels, low_memory=False, delimiter=' ', header=None)
data_test = pd.read_csv(datadir + dataset_test, low_memory=False, delimiter=' ', header=None)
data_test_labels = pd.read_csv(datadir + dataset_test_labels, low_memory=False, delimiter=' ', header=None)
data_train = data_train.iloc[:, :-1]
data_test = data_test.iloc[:, :-1]
[n_samples, n_features] = data_train.shape

# data_train_labels = pd.DataFrame([data_train_labels], columns=[n_features])
# data_test_labels = pd.DataFrame([data_test_labels], columns=[n_features])
train = pd.concat([data_train, data_train_labels], axis=1)
test = pd.concat([data_test, data_test_labels], axis=1)
features = list(range(n_features))

train -= train.min()
train /= train.max()

test = train[:500]

# test -= test.min()
# test /= test.max()
'''


n_wl_trees, _ = get_weaklearner_number(n_features, prob_threshold)

for windows_size_index, windows_size in enumerate(windows_size_array):
    for hybrid_power_threshold_index, hybrid_power_threshold in enumerate(hybrid_power_threshold_array):
        hybrid = []
        vfdt = []
        rf = []
        for trn in range(test_numbers):
            wl_trees = []
            wl_trees_features = get_weaklearner_feature_indexes(n_wl_trees, features)
            for i in wl_trees_features:
                wl_trees.append(Vfdt(list(itemgetter(*i)(features)), delta=0.01, nmin=100, tau=0.5))
            tree = Vfdt(features, delta=0.01, nmin=100, tau=0.5)

            for sample in train.iterrows():
                x = sample[1][:-1]
                y = sample[1][-1]
                tree.update(x, y)  # fit data
                for wl_tree, wl_tree_feature in zip(wl_trees, wl_trees_features):
                    wl_tree.update(itemgetter(*wl_tree_feature)(x), y)

            print('training number {} is done for windows size {} and hybrid power {}.'.format(trn + 1, windows_size,
                                                                                               hybrid_power_threshold))

            hybrid_power = collections.deque(maxlen=windows_size)
            for i in range(windows_size):
                hybrid_power.append(1)


            wl_performance = np.zeros((n_wl_trees, test.shape[0]))
            tree_performance = np.zeros((1, test.shape[0]))


            def update_n_test(x, y, counter):
                y_pred_tree = tree.predict([x])
                y_pred_wl = []
                for wl_tree, wl_tree_feature in zip(wl_trees, wl_trees_features):
                    y_pred_wl.append(wl_tree.predict(np.asarray([itemgetter(*wl_tree_feature)(i) for i in [x]])))
                y_pred_wl = np.asarray(y_pred_wl)

                if y_pred_tree == y: tree_performance[0, counter] = 1
                for c_wl, wl in enumerate(y_pred_wl):
                    if wl == y: wl_performance[c_wl, counter] = 1

                if sum(hybrid_power) > (hybrid_power_threshold * windows_size):
                    y_pred = get_predictions(np.asarray(y_pred_tree).reshape(-1, 1), y_pred_wl)
                else:
                    y_pred = y_pred_tree

                y_pred_wl = get_majority(y_pred_wl.T)
                if y_pred_wl == y:
                    hybrid_power.append(1)
                else:
                    hybrid_power.append(0)
                tree.update(x, y)  # fit data
                for wl_tree, wl_tree_feature in zip(wl_trees, wl_trees_features):
                    wl_tree.update(itemgetter(*wl_tree_feature)(x), y)

                return y_pred, y_pred_tree, get_majority(y_pred_wl)


            # x_test = train[:, :-1]
            # y_test = train[:, -1]
            # y_pred = tree.predict(x_test)
            # y_pred_wl = []
            # for wl_tree, wl_tree_feature in zip(wl_trees, wl_trees_features):
            #     y_pred_wl.append(wl_tree.predict(np.asarray([itemgetter(*wl_tree_feature)(i) for i in x_test])))
            # y_pred_wl = np.asarray(y_pred_wl)
            #
            # y_pred_hybrid = get_predictions(np.asarray(y_pred).reshape(-1, 1).T, y_pred_wl)
            # y_pred_wl = get_majority(y_pred_wl.T)
            # print('Training set:', test.shape[0],end=', \n')
            # print('ACCURACY for Simple: {:.4%}'.format(accuracy_score(y_test, y_pred)))
            # print('ACCURACY for Random Forest: {:.4%}'.format(accuracy_score(y_test, y_pred_wl)))
            # print('ACCURACY for Hybrid Majority: {:.4%}'.format(accuracy_score(y_test, y_pred_hybrid)))

            # x_test = train.values[:2000, :-1]
            # y_test = train.values[:2000, -1]
            # y_pred = tree.predict(x_test)
            # y_pred_wl = []
            # for wl_tree, wl_tree_feature in zip(wl_trees, wl_trees_features):
            #     y_pred_wl.append(wl_tree.predict(np.asarray([itemgetter(*wl_tree_feature)(i) for i in x_test])))
            # y_pred_wl = np.asarray(y_pred_wl)
            # y_pred_hybrid = get_predictions(np.asarray(y_pred).reshape(-1, 1).T, y_pred_wl)
            # y_pred_wl = get_majority(y_pred_wl.T)
            # print('Training set:', test.shape[0],end=', \n')
            # print('ACCURACY for Simple: {:.4%}'.format(accuracy_score(y_test, y_pred)))
            # print('ACCURACY for Random Forest: {:.4%}'.format(accuracy_score(y_test, y_pred_wl)))
            # print('ACCURACY for Hybrid Majority: {:.4%}'.format(accuracy_score(y_test, y_pred_hybrid)))

            counter = 0
            yy = [];yt = [];yw = []
            for row in test.iterrows():
                [a, b, c] = update_n_test(row[1][:-1].__array__(), row[1][-1], counter)
                yy.append(a);yt.append(b);yw.append(c)
                counter += 1

            counter = 0
            for i, j in zip(yy, test.values[:, -1]):
                if i[0] == j:
                    counter += 1
            # print("Hybrid", counter)
            hybrid.append(100 * counter / test.shape[0])

            counter = 0
            for i, j in zip(yt, test.values[:, -1]):
                if i[0] == j:
                    counter += 1
            # print("Tree", counter)
            vfdt.append(100 * counter / test.shape[0])

            counter = 0
            for i, j in zip(yw, test.values[:, -1]):
                if i[0] == j:
                    counter += 1
            # print("Random Forest", counter)
            rf.append(100 * counter / test.shape[0])

        results_acc[:, windows_size_index, hybrid_power_threshold_index] = [np.mean(hybrid), np.mean(vfdt), np.mean(rf)]
        results_std[:, windows_size_index, hybrid_power_threshold_index] = [np.std(hybrid), np.std(vfdt), np.std(rf)]
[_, i, j] = results_acc.shape


if report:
    for iidx in range(i):
        for jidx in range(j):
            print("*"*20)
            print('Results for windwos size = {} and hybrid impact Threshold = {}'
                  .format(windows_size_array[iidx], hybrid_power_threshold_array[jidx]))
            acc = results_acc[:, iidx, jidx]
            print("Accuracy Results")
            print("Hybrid: %", acc[0])
            print("VFDT: %", acc[1])
            print("Random Forest: %", acc[2])

            std = results_std[:, iidx, jidx]
            print("Standard Deviation Results")
            print("Hybrid: ", std[0])
            print("VFDT: ", std[1])
            print("Random Forest: ", std[2])
            print("*"*20+'\n\n')


if plot:
    import matplotlib.pyplot as plt
    for jidx in range(j):
        plt.figure(jidx)
        plt.grid('on')
        plt.title('Scores for Hybrid Impact Threshold {}'.format(hybrid_power_threshold_array[jidx]))
        ind = np.arange(i)
        width = 0.35
        p1 = plt.bar(ind, results_acc[1, :, jidx], width, yerr=results_std[1, :, jidx])
        p2 = plt.bar(ind, results_acc[0, :, jidx] - results_acc[1, :, jidx], width, yerr=results_std[0, :, jidx],
                     bottom=results_acc[1, :, jidx])
        plt.xticks(ind, windows_size_array)
        plt.ylim((60, 100))
        plt.xlabel('Windows Size')
        plt.ylabel('Accuracy')
        plt.legend((p1[0], p2[0]), ('VFDT', 'Hybrid'))
        plt.savefig(figsave_directory + dataset_fig_dir + "/stackedbars_hyi-{}.svg".format(hybrid_power_threshold_array[jidx]), format='svg')
        plt.savefig(figsave_directory + dataset_fig_dir + "/stackedbars_hyi-{}.png".format(hybrid_power_threshold_array[jidx]), format='png')

    for iidx in range(i):
        plt.figure(j + iidx)
        plt.grid('on')
        plt.title('Scores for Windows Size {}'.format(windows_size_array[iidx]))
        ind = np.arange(j)
        width = 0.35
        p1 = plt.bar(ind, results_acc[1, iidx, :], width, yerr=results_std[1, iidx, :])
        p2 = plt.bar(ind, results_acc[0, iidx, :] - results_acc[1, iidx, :], width, yerr=results_std[0, iidx, :],
                     bottom=results_acc[1, iidx, :])
        plt.xticks(ind, hybrid_power_threshold_array)
        plt.ylim((60, 100))
        plt.xlabel('Hybrid Impact Threshold')
        plt.ylabel('Accuracy')
        plt.legend((p1[0], p2[0]), ('VFDT', 'Hybrid'))
        plt.savefig(figsave_directory + dataset_fig_dir + "/stackedbars_wsz-{}.svg".format(windows_size_array[iidx]), format='svg')
        plt.savefig(figsave_directory + dataset_fig_dir + "/stackedbars_wsz-{}.png".format(windows_size_array[iidx]), format='png')

    for iidx in range(i):
        plt.figure(j + i + iidx)
        plt.grid('on')
        plt.title('Scores for Windows Size {}'.format(windows_size_array[iidx]))
        index = np.arange(j)
        bar_width = 0.25
        opacity = 0.8

        rects1 = plt.bar(index, results_acc[0, iidx, :], bar_width,
                         alpha=opacity,
                         label='Hybrid')

        rects2 = plt.bar(index + bar_width, results_acc[1, iidx, :], bar_width,
                         alpha=opacity,
                         label='VFDT')

        rects3 = plt.bar(index + 2 * bar_width, results_acc[2, iidx, :], bar_width,
                         alpha=opacity,
                         label='Random Forest')

        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.xticks(index + 2 * bar_width, hybrid_power_threshold_array)
        plt.legend()
        plt.ylim((60, 100))

        plt.tight_layout()
        plt.savefig(figsave_directory + dataset_fig_dir + "/comparison_wsz-{}.svg".format(windows_size_array[iidx]), format='svg')
        plt.savefig(figsave_directory + dataset_fig_dir + "/comparison_wsz-{}.png".format(windows_size_array[iidx]), format='png')

    for jidx in range(j):
        plt.figure(j + 2*i + jidx)
        plt.grid('on')
        plt.title('Scores for Hybrid Impact Threshold {}'.format(hybrid_power_threshold_array[jidx]))
        index = np.arange(j)
        bar_width = 0.25
        opacity = 0.8

        rects1 = plt.bar(index, results_acc[0, :, jidx], bar_width,
                         alpha=opacity,
                         label='Hybrid')

        rects2 = plt.bar(index + bar_width, results_acc[1, :, jidx], bar_width,
                         alpha=opacity,
                         label='VFDT')

        rects3 = plt.bar(index + 2 * bar_width, results_acc[2, :, jidx], bar_width,
                         alpha=opacity,
                         label='Random Forest')

        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.xticks(index + 2 * bar_width, windows_size_array)
        plt.legend()
        plt.ylim((60, 100))

        plt.tight_layout()
        plt.savefig(figsave_directory + dataset_fig_dir + "/comparison_hyi-{}.svg".format(hybrid_power_threshold_array[jidx]), format='svg')
        plt.savefig(figsave_directory + dataset_fig_dir + "/comparison_hyi-{}.png".format(hybrid_power_threshold_array[jidx]), format='png')


    plt.show()
