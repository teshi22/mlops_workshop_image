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
    # データパスの処理
    train_list = make_datapath_list(train_data_path)
    val_list = make_datapath_list(val_data_path)

    # 画像変換+データセット作成
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = HymenopteraDataset(
        file_list=train_list,
        transform=ImageTransform(size, mean, std),
        phase="train",
    )
    val_dataset = HymenopteraDataset(
        file_list=val_list,
        transform=ImageTransform(size, mean, std),
        phase="val",
    )

    # 動作確認
    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])
    # ミニバッチのサイズを指定
    batch_size = 32

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict


def create_network():
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.train()
    return net


def set_optimizer():
    params_to_update = []
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    return optimizer


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # epochのループ
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-------------")

        # epochごとの学習と検証のループ
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == "train"):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in dataloaders_dict[phase]:

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, torch.tensor(labels))  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(
                dataloaders_dict[phase].dataset
            )

            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc
                )
            )

            mlflow.log_metric(phase + "_loss", epoch_loss, step=epoch)
            mlflow.log_metric(phase + "_acc", epoch_acc, step=epoch)
    return net


def register_model(net, run):
    model_name = "model.pth"
    local_path = f"./{model_name}"
    torch.save(net.state_dict(), local_path)
    mlflow.log_artifact(local_path, "output/")
    run_id = run.info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/output/{model_name}",
        name="transfer_learning_model",
    )
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
