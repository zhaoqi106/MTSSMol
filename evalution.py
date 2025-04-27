import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from dataset.dataset_test_1 import MolTestDatasetWrapper
from models.ginet_finetune import GINet
import pandas as pd
import torch.nn.functional as F

def load_model(config, model_path, device):
    """
    加载预训练模型。
    """
    model = GINet(config['dataset']['task'], **config["model"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def test_model(model, data_loader, device):

    predictions,name, smiles ,features,labels = [], [], [], [],[]

    with torch.no_grad():
        for bn, data in enumerate(data_loader):
            data = data.to(device)

            h, output = model(data)

            predictions.extend(output.cpu().detach().numpy())
            name.extend(data.name)

            smiles.extend(data.smile)
            labels.extend(data.y)
            features.append(h.cpu())

    predictions = np.array(predictions)
    name = name
    smiles = smiles
    labels = labels
    features = torch.cat(features, dim=0)
    print(len(features))

    if len(predictions) != len(name) or len(predictions) != len(smiles) or len(predictions) != len(features):
        raise ValueError("Length of predictions, names, and smiles does not match.")

    df = pd.DataFrame({
        'name': name,
        'smiles': smiles,
        'Prediction': predictions[:, 0],
        'Label': labels
    })

    # 将 features 添加到 DataFrame 中
    feature_cols = {f'feature_{i}': features[:, i].cpu().numpy() for i in range(features.shape[1])}
    df = df.assign(**feature_cols)

    # 将 DataFrame 转换为字典
    df_dict = df.to_dict(orient='list')
    df.to_csv(
        'E:/FDA_BD_Tears/roc/feature_FGFR.csv',
        mode='a', index=False,
    )

    return df_dict


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = yaml.load(open("test_finetune.yaml", "r"), Loader=yaml.FullLoader)
    config['dataset']['task'] = 'regression'

    data_path = 'data/FGFR1_IC50.csv'
    test_dataset_wrapper = MolTestDatasetWrapper(data_path=data_path)
    test_loader = test_dataset_wrapper.get_data_loaders()

    # 模型路径，根据实际情况替换为你的模型存储路径
    model_path = '/checkpoints/model.pth'

    model = load_model(config,model_path, config["gpu"])
    predictions = test_model(model, test_loader, device)
