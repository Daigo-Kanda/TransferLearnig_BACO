import argparse
import copy
import datetime
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

import ITrackerData_person_tensor as data_gen
from BinaryAntColonyOptimization import BinaryAntColonyOptimization


def print_varsize():
    import types
    print("{}{: >15}{}{: >10}{}".format('|', 'Variable Name', '|', '  Size', '|'))
    print(" -------------------------- ")
    for k, v in globals().items():
        if hasattr(v, 'size') and not k.startswith('_') and not isinstance(v, types.ModuleType):
            print("{}{: >15}{}{: >10}{}".format('|', k, '|', str(v.size), '|'))
        elif hasattr(v, '__len__') and not k.startswith('_') and not isinstance(v, types.ModuleType):
            print("{}{: >15}{}{: >10}{}".format('|', k, '|', str(len(v)), '|'))


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def gene2model_aco(model, dic_path):
    for layer in model.layers:
        # 重みがあればgeneを参照してtrainableを変更
        #   新規学習レイヤーの場合は重みをロード
        if layer.name in dic_path:
            layer.trainable = bool(dic_path[layer.name])


def main(args):
    core_model = keras.models.load_model(args.trained_model)

    weights = core_model.get_weights()

    # make baco instance
    baco = BinaryAntColonyOptimization(core_model)

    # algorithm cycle
    cycle = 20
    n_geopath = 20

    # generate data
    data = data_gen.getData(batch_size=args.batch_size, memory_size=150, dataset_path=args.dataset_dir)
    train_generator = data[0]
    validation_generator = data[1]

    # save best val_loss model
    best_val_loss = 1000

    print(args.save_dir)
    if not os.path.exists(args.save_dir):
        print(args.save_dir)
        os.makedirs(args.save_dir)

    # ログ用
    history_log = []
    geopath_best_log = []

    for i in range(cycle):
        li_path = []
        li_edge = []
        li_loss = []
        histories = []

        # make path and that edge
        for j in range(n_geopath):
            tmp = baco.gen_path()
            for k in tmp[0]:
                if 'dense' in k:
                    tmp[0][k] = 1
            li_path.append(tmp[0])
            li_edge.append(tmp[1])

        for k in range(n_geopath):
            get_model_time = time.time()
            # make model from path data (set freeze or not)

            copy_model = keras.models.clone_model(core_model)
            copy_model.set_weights(weights)

            output = copy_model(copy_model.inputs, training=False)
            model = keras.Model(copy_model.inputs, output)

            gene2model_aco(copy_model, li_path[k])

            model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
            # print(model.optimizer.get_weights())

            print('model prepare need {} seconds'.format(time.time() - get_model_time))

            fit_time = time.time()

            # train just one epoch
            history = model.fit(
                x=train_generator,
                initial_epoch=0,
                epochs=1,
                verbose=2,
                validation_data=validation_generator,
            )

            print('fit time is {}'.format(time.time() - fit_time))

            # results
            history = np.array([
                [history.history['mae'][-1], history.history['loss'][-1], history.history['val_mae'][-1],
                 history.history['val_loss'][-1]]
            ])

            # save acc
            li_loss.append(history[0, 3])
            histories.append(copy.deepcopy(history))

            tf.keras.backend.clear_session()
            # ガベコレ
            del history, model, copy_model, output
            gc.collect()

        # print('elapsed time is {}'.format(time.time() - start))

        # 結果の集計
        tmp_loss = [history[0, 3] for history in histories]
        which_win = np.argmin(tmp_loss)

        if tmp_loss[which_win] <= best_val_loss:
            best_val_loss = tmp_loss[which_win]
            best_path = li_path[which_win]
            print(best_path)

        # ログの格納
        history_log.append(histories[which_win])
        geopath_best_log.append(copy.deepcopy(li_path[which_win]))
        # geopath_set_log.append(copy.deepcopy(li_path))

        # update pheromone for next cycle
        baco.update_pheromone(li_edge, li_loss)

        # ガベコレ
        del histories, li_path, li_edge, li_loss, tmp_loss, which_win
        gc.collect()

    # ログデータの吐き出し
    data = np.array(history_log).reshape((cycle, 4))
    df_data = pd.DataFrame(data, columns=['mae', 'loss', 'val_mae', 'val_loss'])
    df_data['geopath_best'] = [json.dumps(x, default=myconverter) for x in geopath_best_log]

    now = datetime.datetime.now()
    epoch_log_name = 'swpathnet_%s-tournament-%s.csv' % ('baco', now.strftime('%Y%m%d_%H%M%S'))

    df_data.to_csv(os.path.join(args.save_dir, epoch_log_name), header=True)

    ####################################################################################################################
    # train best path
    copy_model = keras.models.clone_model(core_model)
    copy_model.set_weights(weights)

    output = copy_model(copy_model.inputs, training=False)
    model = keras.Model(copy_model.inputs, output)

    gene2model_aco(copy_model, best_path)

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])

    now = datetime.datetime.now()
    # callbacks
    cbks = [keras.callbacks.CSVLogger(
        os.path.join(args.save_dir, '%s_%s_%s.csv' % ('baco', "eye_tracking", now.strftime('%Y%m%d_%H%M%S')))),
        ModelCheckpoint(
            os.path.join(args.save_dir, "models.{}.hdf5".format(now.strftime('%Y%m%d_%H%M%S'))),
            monitor='val_loss', save_best_only=True,
            save_weights_only=False)
    ]

    model.fit(
        x=train_generator,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        validation_data=validation_generator,
        callbacks=cbks,
    )

    ####################################################################################################################

    tf.keras.backend.clear_session()


def get_parser():
    parser = argparse.ArgumentParser()

    # データセットのディレクトリ
    parser.add_argument('dataset_dir')

    # ログの保存先ディレクトリ
    parser.add_argument('save_dir')

    # 学習データの画像数
    #   CIFAR10: 50000, CIFAR100: 50000
    parser.add_argument('--num_images_train', type=int, default=50000, help='CIFAR10: 50000, CIFAR100: 50000')

    # テストデータの画像数
    #   CIFAR10: 10000, CIFAR100: 10000
    parser.add_argument('--num_images_test', type=int, default=10000, help='CIFAR10: 10000, CIFAR100: 10000')

    # 画像サイズ
    parser.add_argument('--image_size', type=int, default=224)

    # 遺伝子型の数
    parser.add_argument('--n_geopath', type=int, default=20)

    # 世代数
    parser.add_argument('--epochs', type=int, default=30)

    # 勝負する遺伝子型の数
    parser.add_argument('--n_comp', type=int, default=2)

    # バッチサイズ
    parser.add_argument('--batch_size', type=int, default=16)

    # 1世代で学習するエポック数
    parser.add_argument('--geopath_epochs', type=int, default=1)

    # GPU並列
    parser.add_argument('--n_gpu', type=int, default=1)

    # CPU並列
    parser.add_argument('--n_thread', type=int, default=1)

    # CPU(スレッド)並列
    #   fit_generatorでスレッド並列するとデッドロックする臭い？
    #   https://github.com/keras-team/keras/issues/10340
    parser.add_argument('--use_multiprocessing', action='store_true')

    # 水増しの有無
    parser.add_argument('--dont_augment', action='store_false')

    # 層の範囲の選択
    #   True: 全て（含むBN）, False: CNN, FCのみ
    parser.add_argument('--transfer_all', action='store_true')

    # 学習可能層の初期値に学習済みパラメータを使う
    parser.add_argument('--finetune', action='store_true')

    # （テスト用）学習済みモデルの設定
    parser.add_argument('--trained_model', default=None)

    # 使う学習済みモデル
    parser.add_argument('--model_name', default='vgg16',
                        help='name of pre-trained network. this is disable by giving --trained_model')
    # Flag for original or proposed
    parser.add_argument('--do_original', action='store_true')

    return parser


# 引数の読み込み
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print('----PARSED ARGS----\n%s\n-----------------' % args)

    main(args)
