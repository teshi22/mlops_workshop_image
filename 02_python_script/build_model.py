import mlflow
from azureml.core import Workspace
import argparse
import glob
import os.path as osp

from PIL import Image
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train_data_dir",
        type=str,
        help="root directory of train folders",
        default="../data/hymenoptera_data/train",
    )
    parser.add_argument(
        "--input_valid_data_dir",
        type=str,
        help="root directory of val folders",
        default="../data/hymenoptera_data/val",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "remote"],
        help="execution mode",
        default="local",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    return args


class HymenopteraDataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'val'
        訓練か検証かを設定
    """

    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        """画像の枚数を返す"""
        return len(self.file_list)

    def __getitem__(self, index):
        """
        前処理をした画像のTensor形式のデータとラベルを取得
        """

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase
        )  # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        if self.phase == "train":
            if "ants" in img_path:
                label = 0
            elif "bees" in img_path:
                label = 1
        elif self.phase == "val":
            if "ants" in img_path:
                label = 0
            elif "bees" in img_path:
                label = 1

        return img_transformed, label


def make_datapath_list(dataset_path: str):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    target_path = osp.join(dataset_path) + "/**/*.jpg"
    print(target_path)

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class ImageTransform:
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        resize, scale=(0.5, 1.0)
                    ),  # データオーギュメンテーション
                    transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                    transforms.ToTensor(),  # テンソルに変換
                    transforms.Normalize(mean, std),  # 標準化
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),  # リサイズ
                    transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                    transforms.ToTensor(),  # テンソルに変換
                    transforms.Normalize(mean, std),  # 標準化
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)


def load_data(train_data_path: str, val_data_path: str):
    ## 記述 ##
    return ""


def create_network():
    ## 記述 ##
    return ""


def set_optimizer():
    ## 記述 ##
    return ""


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    ## 記述 ##
    return ""


def register_model(net, run):
    ## 記述 ##
    return ""


if __name__ == "__main__":

    args = parse_args()

    # ロカール環境でのmlflowセットアップ
    if args.mode == "local":
        print("mode local")
        subscription_id = "5290deef-ab3d-4e26-90bb-2296ecd99c71"
        resource_group = "ml-handson"
        workspace = "ml-workspace"

        ws = Workspace(
            workspace_name=workspace,
            subscription_id=subscription_id,
            resource_group=resource_group,
        )

        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

        experiment_name = "transfer_learning_job"
        mlflow.set_experiment(experiment_name)

    # メイン処理
    run = mlflow.start_run()

    # データ読み込み
    dataloaders_dict = load_data(
        args.input_train_data_dir, args.input_valid_data_dir
    )

    # ネットワーク作成
    net = create_network()

    # 損失関数を定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法の設定
    optimizer = set_optimizer()

    # モデル学習
    num_epochs = args.epochs
    net = train_model(
        net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs
    )

    # モデル保存
    register_model(net, run)

    mlflow.end_run()
