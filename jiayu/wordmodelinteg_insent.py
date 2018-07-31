# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import pickle
import collections
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pydotplus
# from IPython.display import Image
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import os
import nltk
import requests
from optparse import OptionParser
import time
import Environment as env
from sklearn.externals import joblib

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
np.random.seed(121)


class InputPara:
    def __init__(self, in_dir, out_dir, ba_ratio):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.ba_ratio = ba_ratio


def fpre(pitch, dur):
    pitch = pitch.reset_index()
    pitch.columns = ['frame', 'pitch']
    pitch.frame = pitch.frame + 1
    dur = dur[dur['unit'] != '<sil>']
    dur = dur.reset_index()
    return (pitch, dur)


def gen_one_data_basicfeature(path_f0, path_dur, path_syl, index):
    f0 = pd.read_table(path_f0, sep='\t')
    dur = pd.read_table(path_dur)
    syl = pd.read_table(path_syl)

    (f0, dur) = fpre(f0, dur)
    f0_wozero = f0[f0['pitch'] != 0]
    med = f0_wozero['pitch'].median()
    f0_wozero['ome'] = f0_wozero['pitch'].apply(lambda x: math.log((x / med), 2))

    list_pitch_mean = []
    list_pitch_max = []
    list_pitch_min = []
    list_pitch_median = []
    for i in range(len(dur)):
        word_pitch = f0_wozero.ix[dur.iloc[i, 1] - 1:dur.iloc[i, 2] - 1]
        word_pitch_mean = word_pitch[word_pitch['pitch'] != 0].ome.mean()
        word_pitch_max = word_pitch[word_pitch['pitch'] != 0].ome.max()
        word_pitch_min = word_pitch[word_pitch['pitch'] != 0].ome.min()
        word_pitch_median = word_pitch[word_pitch['pitch'] != 0].ome.median()
        list_pitch_mean.append(word_pitch_mean)
        list_pitch_max.append(word_pitch_max)
        list_pitch_min.append(word_pitch_min)
        list_pitch_median.append(word_pitch_median)
        list_pitch_range = list(map(lambda x: x[0] - x[1], zip(list_pitch_max, list_pitch_min)))

    syl_num_lst = []
    for w_start, w_end in zip(dur['start_frame'], dur['end_frame']):
        n = 0
        for s in syl['start_frame']:
            # print(w_start,w_end,s)
            if w_start <= s < w_end:
                n += 1
        syl_num_lst.append(n)

    c = {'pitch_mean': list_pitch_mean, 'pitch_max': list_pitch_max, 'pitch_range': list_pitch_range,
         'syl_num': syl_num_lst,
         'pitch_median': list_pitch_median}
    df = pd.DataFrame(c)

    df = df.reset_index()
    # print(df)
    df['uniq'] = df['index'].apply(lambda x: index + '-' + str(x))
    df['pitch_mean'].fillna(-1, inplace=True)
    df['pitch_max'].fillna(-1, inplace=True)
    df['pitch_range'].fillna(0, inplace=True)
    df['pitch_median'].fillna(0, inplace=True)

    df['word'] = dur['unit']
    df['intensity'] = dur['mean energy']
    df['duration'] = dur['frame num']
    df['duration_norm'] = df['duration'] / df['syl_num']

    for j in range(len(df)):
        if j != 0:
            df.ix[j, 'max_diff'] = df.ix[j, 'pitch_max'] - df.ix[j - 1, 'pitch_max']
            df.ix[j, 'dur_diff'] = df.ix[j, 'duration_norm'] - df.ix[j - 1, 'duration_norm']
            df.ix[j, 'intensity_diff'] = df.ix[j, 'intensity'] - df.ix[j - 1, 'intensity']
        else:
            df.ix[j, 'max_diff'] = 0
            df.ix[j, 'dur_diff'] = 0
            df.ix[j, 'intensity_diff'] = 0

    for h in range(len(df)):
        if h != len(df) - 1:
            # print(h)
            df.ix[h, 'max_diff_after'] = df.ix[h, 'pitch_max'] - df.ix[h + 1, 'pitch_max']
            df.ix[h, 'dur_diff_after'] = df.ix[h, 'duration_norm'] - df.ix[h + 1, 'duration_norm']
            df.ix[h, 'intensity_diff_after'] = df.ix[h, 'intensity'] - df.ix[h + 1, 'intensity']
        else:
            df.ix[h, 'max_diff_after'] = 0
            df.ix[h, 'dur_diff_after'] = 0
            df.ix[h, 'intensity_diff_after'] = 0

    df['pitch_max_rank'] = df['pitch_max'].rank(ascending=False)
    df['pitch_mean_rank'] = df['pitch_mean'].rank(ascending=False)
    df['pitch_range_rank'] = df['pitch_range'].rank(ascending=False)
    df['pitch_median_rank'] = df['pitch_median'].rank(ascending=False)
    df['max_diff_rank'] = df['max_diff'].rank(ascending=False)
    df['max_diff_after_rank'] = df['max_diff'].rank(ascending=False)

    df['intensity_rank'] = df['intensity'].rank(ascending=False)
    df['intensity_diff_rank'] = df['intensity_diff'].rank(ascending=False)
    df['intensity_diff_after_rank'] = df['intensity_diff_after'].rank(ascending=False)
    df['intensity_diff_rank_per'] = df['intensity_diff_rank'] / len(df)
    df['intensity_diff_after_rank_per'] = df['intensity_diff_after_rank'] / len(df)

    df['duration_norm_rank'] = df['duration_norm'].rank(ascending=False)
    df['dur_diff_rank'] = df['dur_diff'].rank(ascending=False)
    df['dur_diff_after_rank'] = df['dur_diff_after'].rank(ascending=False)
    df['dur_diff_rank_per'] = df['dur_diff_rank'] / len(df)
    df['dur_diff_after_rank_per'] = df['dur_diff_after_rank'] / len(df)

    df['pitch_max_rank_per'] = df['pitch_max_rank'] / len(df)
    df['pitch_mean_rank_per'] = df['pitch_mean_rank'] / len(df)
    df['pitch_range_rank_per'] = df['pitch_range_rank'] / len(df)
    df['pitch_median_rank_per'] = df['pitch_median_rank'] / len(df)
    df['max_diff_rank_per'] = df['max_diff_rank'] / len(df)
    df['max_diff_after_rank_per'] = df['max_diff_after_rank'] / len(df)
    df['intensity_rank_per'] = df['intensity_rank'] / len(df)
    df['duration_norm_rank_per'] = df['duration_norm_rank'] / len(df)
    return df


def gen_basicfeature(folder):
    frame = []
    for file in os.listdir(folder):
        index = file[6:]
        path_f0 = os.path.join(folder, file, 'align_out_s', index + '.f0')
        path_dur = os.path.join(folder, file, 'align_out_s', index + '.word.mean.align')
        path_syl = os.path.join(folder, file, 'align_out_s', index + '.syllable.mean.align')
        # path_phone = os.path.join(r'E:\stress_result\result_extract_400',file,'align_out_s',index+'.phone.mean.align')
        df_ = gen_one_data_basicfeature(path_f0, path_dur, path_syl, index)
        frame.append(df_)
    data_feature_wopos = pd.concat(frame)
    return data_feature_wopos


def gen_pos(ser_text, ser_index):
    frame = []
    for i, ind in zip(ser_text, ser_index):
        i = i.replace("*", "")
        rep = requests.post(env.stanford_url, data=i)
        pos_data = []
        for sen in range(len(rep.json()['sentences'])):
            pos_data.extend(rep.json()['sentences'][sen]['tokens'])

        pos = pd.DataFrame(pos_data)

        pos['isalpha'] = pos['word'].apply(lambda x: x.isalpha())
        for i in range(len(pos)):
            if i == 0:
                pos.ix[i, 'pos_before'] = 'none'
                pos.ix[i, 'pos_after'] = pos.ix[i + 1, 'pos']
            if i == len(pos) - 1:
                pos.ix[i, 'pos_after'] = 'none'
                pos.ix[i, 'pos_before'] = pos.ix[i - 1, 'pos']
            elif i != 0 and i != len(pos) - 1:
                pos.ix[i, 'pos_before'] = pos.ix[i - 1, 'pos']
                pos.ix[i, 'pos_after'] = pos.ix[i + 1, 'pos']

        pos = pos[pos['isalpha'] == True]
        pos['index_'] = ind
        pos = pos.reset_index()
        pos = pos.drop(['level_0'], axis=1)
        pos = pos.reset_index()
        pos['uniq'] = pos['level_0'].apply(lambda x: ind + '-' + str(x))
        frame.append(pos)
    pos_df = pd.concat(frame)
    return pos_df


# def gen_pos_onehot(data_wpos):
#     one_hot_pos = pd.get_dummies(data_wpos['pos'])
#     one_hot_pos_before = pd.get_dummies(data_wpos['pos_before'], prefix='before')
#     one_hot_pos_after = pd.get_dummies(data_wpos['pos_after'], prefix='after')
#     data_wpos_wonthot = pd.concat([data_wpos, one_hot_pos, one_hot_pos_before, one_hot_pos_after], axis=1)
#     return data_wpos_wonthot


def gen_onehot(pos_series, posname):
    frame = []
    for i in pos_series:
        init = [0] * len(posname)
        index = posname.index(i)
        init[index] = 1
        dic = collections.OrderedDict(zip(posname, init))
        onehot_df = pd.DataFrame.from_dict(dic, orient='index').T
        frame.append(onehot_df)
        # print('-------')
    data_pos = pd.concat(frame)
    data_pos = data_pos.reset_index()
    data_pos.drop('index', axis=1, inplace=True)
    # data = pd.concat([data_pos,df_],axis = 1)
    return data_pos


def gen_label(ser_text, ser_index):
    frame2 = []
    for text, index in zip(ser_text, ser_index):
        word = text.split(' ')
        sen_df = pd.DataFrame({'word': word})
        sen_df = sen_df.reset_index()
        sen_df['uniq'] = sen_df['index'].apply(lambda x: index + '-' + str(x))
        sen_df['label'] = sen_df['word'].apply(lambda x: x.count('*'))
        frame2.append(sen_df)
    label_concat = pd.concat(frame2)
    return (label_concat)


def balance_sample(data, balance_ratio):
    num_stress = len(data[data.label == 1])
    stress_indices = np.array(data[data.label == 1].index)
    nonstress_indices = data[data.label == 0].index
    random_nonstress_indices = np.random.choice(nonstress_indices, balance_ratio * num_stress, replace=False)
    random_nonstress_indices = np.array(random_nonstress_indices)
    under_sample_indices = np.concatenate([stress_indices, random_nonstress_indices])
    under_sample_data = data.ix[under_sample_indices, :]
    x = under_sample_data.ix[:, under_sample_data.columns != 'label']
    target = under_sample_data.ix[:, under_sample_data.columns == 'label']
    return (x, target)


def train_model(X_train, y_train):
    dtr = tree.DecisionTreeClassifier()
    # cv_scores_dtr = cross_val_score(dtr, X_train, y_train, cv=5, scoring='precision')
    # dtr.fit(X_train, y_train)
    # original_score_dtr = dtr.score(X_test, y_test)
    # print('DecisionTree cross validation result(precision):')
    # print(cv_scores_dtr)
    # print(cv_scores_dtr.mean())
    # print('original model test result(R2):')
    # print(original_score_dtr)

    tree_param_grid2 = {'max_depth': list((3, 4, 5, 6, 7, 8, 9, 10)),
                        'min_samples_split': list((3, 4, 5, 6, 7, 8, 9, 10))}
    grid2 = GridSearchCV(dtr, param_grid=tree_param_grid2, cv=5, scoring='precision')
    grid2.fit(X_train, y_train)
    # print(grid2.scorer_)
    dtr_best_para = grid2.best_params_
    # print(dtr_best_para['min_samples_split'])
    # print(dtr_best_para['max_depth'])
    # print(grid2.best_params_)
    # print(grid2.best_score_)

    dtr = tree.DecisionTreeClassifier(max_depth=dtr_best_para['max_depth'],
                                      min_samples_split=dtr_best_para['min_samples_split'])
    dtr.fit(X_train, y_train)
    # best_score_dtr = dtr.score(X_test, y_test)
    # print('best parameter result(dtr)')
    # print(best_score_dtr)
    #     from sklearn.metrics import classification_report

    #     y_pred = dtr.predict(X_test)
    #     target_names = ['non_stress', 'stress']
    #     print(classification_report(y_test, y_pred, target_names=target_names))

    # print("RandomForestClassifier  Result:")

    # rfr = RandomForestClassifier()
    # rfr.fit(X_train, y_train)
    # original_score_rfr = rfr.score(X_test, y_test)

    # cv_scores_rfr = cross_val_score(rfr, X_train, y_train, cv=5, scoring='precision')
    # print('RandomForest cross validation result(precision):')
    # print(cv_scores_rfr)
    # print(cv_scores_rfr.mean())
    # print('original model test result(R2):')
    # print(original_score_rfr)

    tree_param_grid = {'min_samples_split': list((3, 4, 5, 6, 7, 8, 9)),
                       'n_estimators': list((10, 20, 30, 40, 50, 60, 70, 80, 90, 100))}
    grid = GridSearchCV(RandomForestClassifier(), param_grid=tree_param_grid, cv=5, scoring='precision')
    grid.fit(X_train, y_train)
    # print(grid.scorer_)
    rfr_best_para = grid.best_params_

    # print(rfr_best_para['min_samples_split'])
    # print(rfr_best_para['n_estimators'])
    print(grid.best_params_)
    print(grid.best_score_)

    rfr = RandomForestClassifier(min_samples_split=rfr_best_para['min_samples_split'],
                                 n_estimators=rfr_best_para['n_estimators'])
    cv_scores_rfr = cross_val_score(rfr, X_train, y_train, cv=5, scoring='precision')
    print('RandomForest cross validation result(precision):')
    print(cv_scores_rfr)
    print(cv_scores_rfr.mean())
    rfr.fit(X_train, y_train)
    # best_score_rfr = rfr.score(X_test, y_test)
    print('best parameter result(rfr)')
    # print(best_score_rfr)

    y_train_pred = rfr.predict(X_train)
    target_names = ['non_stress', 'stress']
    train_report = classification_report(y_train, y_train_pred, target_names=target_names)

    feature_importance = pd.DataFrame({'feature': list(X_train.columns), 'importances': rfr.feature_importances_})
    feature_importance = feature_importance.sort_index(by='importances', ascending=False)
    print(feature_importance)
    return (feature_importance, train_report, dtr, rfr)


def test_model(X_test, y_test, rfr):
    y_pred = rfr.predict(X_test)
    # y_train_pred = rfr.predict(X_train)
    target_names = ['non_stress', 'stress']
    # print('report')
    test_report = classification_report(y_test, y_pred, target_names=target_names)
    # print(test_report)
    # train_report = classification_report(y_train, y_train_pred, target_names=target_names)
    # print(train_report)

    test_confusion_matrix = confusion_matrix(y_test, y_pred)
    # print(test_confusion_matrix)
    return (test_report, test_confusion_matrix)


# def plot_tree(dtr,x):
#     dot_data = tree.export_graphviz(
#         dtr,
#         out_file=None,
#         feature_names=x.columns,
#         filled=True,
#         impurity=False,
#         rounded=True
#     )
#
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.get_nodes()[7].set_fillcolor("#FFF2DD")
#     graph.write_png("dtr_white_background.png")
#     image = Image(graph.create_png())
#     return image


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.draw()
    plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def stress_detect():
    # raw_data_list = os.listdir(in_para.in_dir)
    # raw_data_dir = os.path.join(in_para.in_dir, raw_data_list[0])
    # stress_label = os.path.join(raw_data_dir, "all.xlsx")
    # raw_data_list = os.listdir(in_para.in_dir)
    raw_data_dir = os.path.join(in_para.in_dir, 'data1')
    data_input_folder = os.path.join(raw_data_dir, 'result_extract_400')
    stress_label = os.path.join(raw_data_dir, "insent4.xlsx")
    # raw_data_dir = in_para.in_dir
    # stress_label = in_para.label_file
    balance_ratio = int(in_para.ba_ratio)

    # input
    # raw_data_dir = r'E:\stress_result\result_extract_400'
    # stress_label = r'E:\stress_label\all.xlsx'
    # balance_ratio = 1

    # x feature extraction
    data_feature_wopos_ = gen_basicfeature(data_input_folder)
    sen = pd.read_excel(stress_label, header=None, names=['text', 'index'])
    sen['index'] = sen['index'].apply(lambda x: str(x).zfill(10))
    pos_df_ = gen_pos(sen['text'], sen['index'])
    data_wpos_ = pd.merge(pos_df_, data_feature_wopos_, on='uniq')

    pos_lst = []
    f = open('pos', 'r')
    a = f.readlines()
    for i in a:
        pos_lst.append(i.split('\n')[0])
    f.close()
    pos_lst.append('none')

    before_pos_lst = ['before_' + x for x in pos_lst]
    after_pos_lst = ['after_' + x for x in pos_lst]
    data_pos_ = gen_onehot(data_wpos_['pos'], pos_lst)
    data_pos_before_ = gen_onehot(data_wpos_['pos_before'], pos_lst)
    data_pos_after_ = gen_onehot(data_wpos_['pos_after'], pos_lst)
    data_pos_before_.columns = before_pos_lst
    data_pos_after_.columns = after_pos_lst
    data_wpos_wonthot_ = pd.concat([data_wpos_, data_pos_, data_pos_before_, data_pos_after_], axis=1)

    # y feature extraction
    label = copy.deepcopy(sen)
    label_concat_ = gen_label(label['text'], label['index'])

    # merge x, y for training
    data_ = pd.merge(data_wpos_wonthot_, label_concat_, on='uniq')
    print('before drop features')
    print(data_.columns)
    # drop feature config

    data_.drop(env.feature_drop_list, axis=1, inplace=True)
    feature_name = list(data_.columns)
    print(feature_name)
    # print(data_.columns)
    # balance stress's samples
    x_, target_ = balance_sample(data_, balance_ratio)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(x_, target_, test_size=0.25)
    # load_or_not = True

    # if load_or_not and os.path.isfile(os.path.join(in_para.out_dir,'stress_model.m'):
    #     rfr_ = joblib.load(os.path.join(in_para.out_dir,'stress_model.m')
    feature_importance_, train_report_, dtr_, rfr_ = train_model(X_train_, y_train_)

    test_report_, test_confusion_matrix_ = test_model(X_test_, y_test_, rfr_)
    print(test_report_, train_report_, test_confusion_matrix_)

    # draw decision tree
    # plot_tree(dtr_, X_train_)
    # precision = test_report_.split('non_stress ')[1].split(' ')[0]
    resultfile = time.strftime('%Y-%m-%d-%H%M', time.localtime())
    train_and_test_pickle = 'train_and_test_pickle'
    os.makedirs(os.path.join(in_para.out_dir, resultfile,train_and_test_pickle))
    model_file_path = os.path.join(in_para.out_dir, resultfile, "stress_model.m")
    # model_file_path = os.path.join(in_para.out_dir,"stress_model.m")
    test_report_path = os.path.join(in_para.out_dir, resultfile, "result.txt")
    feature_importance_file_path = os.path.join(in_para.out_dir, resultfile, "feature_importance.xlsx")
    joblib.dump(rfr_, open(model_file_path, 'wb'))

    feature_name_file_path = os.path.join(in_para.out_dir, resultfile, "feature.txt")
    f = open(feature_name_file_path, 'w')
    f.write(str(feature_name))
    f.close()

    feature_importance_.to_excel(feature_importance_file_path)
    f2 = open(test_report_path, 'w')
    f2.write(str(test_report_))
    f2.close()

    # pickle train and test data for adding data
    pic1 = open(os.path.join(in_para.out_dir, resultfile, train_and_test_pickle, 'ori_X_train.pkl'), 'wb')
    pickle.dump(X_train_, pic1)
    pic1.close()
    pic2 = open(os.path.join(in_para.out_dir, resultfile, train_and_test_pickle, 'ori_y_train.pkl'), 'wb')
    pickle.dump(y_train_, pic2)
    pic2.close()
    pic3 = open(os.path.join(in_para.out_dir, resultfile, train_and_test_pickle, 'ori_X_test.pkl'), 'wb')
    pickle.dump(X_test_, pic3)
    pic3.close()
    pic4 = open(os.path.join(in_para.out_dir, resultfile, train_and_test_pickle, 'ori_y_test.pkl'), 'wb')
    pickle.dump(y_test_, pic4)
    pic4.close()

    # model = pickle.load(open(model_file_path, 'rb'), encoding="bytes")

    # learning curve
    title = "Learning Curves"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = rfr_
    plot_learning_curve(estimator, title, X_train_, y_train_, (0.7, 1.01), cv=cv, n_jobs=4)

    print("stress detect finished.")


def main():
    usage = "Usage: Stress detection model training and evaluation.\n " \
            "Parameter: input_dir output_dir balance_ratio"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 3:
        print(usage)
        return

    global in_para
    in_para = InputPara(args[0], args[1], args[2])

    start_time = time.time()
    stress_detect()
    stop_time = time.time()
    print("elapsed time: %fs", stop_time - start_time)


if __name__ == "__main__":
    main()
