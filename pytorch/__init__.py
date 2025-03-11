import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# インポート先での確認用コメント
README = 'Common Library for PyTorch\nAuthor: C. Kuriyama'

# PyTorch乱数固定用


def random_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# モデル学習用


def fit(history, num_epochs, net, optimizer, criterion, train_loader, test_loader, device):
    from tqdm.notebook import tqdm

    # 追加で学習させる際、今までのエポック数を取得する（初めての学習の場合は0となる）
    base_epochs = len(history)

    #
    for epoch in range(base_epochs, base_epochs + num_epochs):
        # 1エポックごとの累計の精度用
        epoch_train_acc, epoch_test_acc = 0, 0

        # 1エポックごとの累計の損失用
        epoch_train_loss, epoch_test_loss = 0, 0

        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        # 訓練フェーズ
        net.train()

        for inputs, labels in tqdm(train_loader):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)

            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size

            # 勾配の初期化
            optimizer.zero_grad()

            # GPUへ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 予測
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ更新
            optimizer.step()

            # 予測ラベル導出
            predicted = torch.max(outputs, 1)[1]

            # 1エポックごとの累計精度の計算
            epoch_train_acc += (predicted == labels).sum().item()

            # 1エポックごとの累積損失の計算
            epoch_train_loss += loss.item() * train_batch_size

        # 予測フェーズ
        net.eval()

        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)

            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUへ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測
            outputs = net(inputs_test)

            # 損失計算
            loss = criterion(outputs, labels_test)

            # 予測ラベル導出
            predicted_test = torch.max(outputs, 1)[1]

            # 1エポックごとの累計精度の計算
            epoch_test_acc += (predicted_test == labels_test).sum().item()

            # 1エポックごとの累積損失の計算
            epoch_test_loss += loss.item() * test_batch_size

        # 精度計算(1データ当たりの平均)
        avg_train_acc = epoch_train_acc / n_train
        avg_test_acc = epoch_test_acc / n_test

        # 損失計算(1データ当たりの平均)
        avg_train_loss = epoch_train_loss / n_train
        avg_test_loss = epoch_test_loss / n_test
        print(f'''エポック [{epoch + 1}/{num_epochs}] 
    【訓練】 損失: {avg_train_loss:.5f} 精度: {avg_train_acc:.5f} 
    【検証】 損失: {avg_test_loss:.5f} 精度: {avg_test_acc:.5f}''')

        item = np.array([epoch + 1, avg_train_loss,
                        avg_train_acc, avg_test_loss, avg_test_acc])
        history = np.vstack((history, item))

    return history


# 学習履歴の表示用
def show_history(history):
    # 献上フェーズの最初のエポックと最後のエポックの記録を取得
    print(f'初期状態: 損失: {history[0, 3]:.5f} 精度: {history[0, 4]:.5f}')
    print(f'最終状態: 損失: {history[-1, 3]:.5f} 精度: {history[-1, 4]:.5f}')

    num_epochs = len(history)
    unit = num_epochs / 10

    plt.plot(history[:, 0], history[:, 1], label='訓練')
    plt.plot(history[:, 0], history[:, 3], label='検証')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    plt.show()

    plt.plot(history[:, 0], history[:, 2], label='訓練')
    plt.plot(history[:, 0], history[:, 4], label='検証')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    plt.show()
