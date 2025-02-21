import argparse
import copy
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models.losses import SupConLoss
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from scripts.utils import batch_generator, plot_embedding, set_random_seed

def get_args():
    # Command setting
    parser = argparse.ArgumentParser(description='Open Set Domain Adaptation')
    parser.add_argument('-model_name', type=str, default='OuterAdapter', help='model name')
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-moment', type=float, default=0.9)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--log-interval', type=int, default=100, help='# batches to wait before logging training status')
    parser.add_argument('-cuda', type=int, default=1, help='cuda id')
    parser.add_argument('-seed', type=int, default=0, help='random seed')

    return parser.parse_args()

def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class OuterAdapterModel(nn.Module):
    def __init__(self, args, num_shared_classes, input_dim, bottleneck_width=32, width=168):
        super(OuterAdapterModel, self).__init__()
        self.args = args
        self.num_shared_classes = num_shared_classes
        self.dim = input_dim
        # Two MLP layers for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.dim, width),
            nn.ReLU(),
            nn.Linear(width, bottleneck_width),
            nn.ReLU(),
        )

        # self.discriminator = nn.Linear(bottleneck_width, 2)
        self.discriminator = nn.Sequential(
            nn.Linear(bottleneck_width, bottleneck_width),
            nn.ReLU(),
            nn.Linear(bottleneck_width, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, outputs, alpha=1):
        feats = []
        for i in range(len(inputs)):
            feats.append(self.feature_extractor(inputs[i]))

        target_idx = outputs[1] <= self.num_shared_classes
        domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([feats[0], feats[1][target_idx]], dim=0), alpha))
        domain_labels = np.array([0] * feats[0].shape[0] + [1] * feats[1][target_idx].shape[0])
        domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=inputs[0].device)
        domain_loss = self.criterion(domain_preds, domain_labels)

        criterion = SupConLoss(temperature=0.07)
        combined_feats = torch.cat([feats[0], feats[1]], dim=0)
        feats_extend = torch.unsqueeze(combined_feats, dim=1)
        feats_extend = F.normalize(feats_extend, dim=-1)
        cl_loss = criterion(feats_extend, torch.cat([outputs[0], outputs[1]]))
        loss = domain_loss*0.5 + cl_loss
        return loss

    def get_feats(self, x, device):
        x = torch.tensor(x, requires_grad=False).to(device)
        x = self.feature_extractor(x).view(x.shape[0], -1)
        return x.cpu().detach().numpy()


#def visualized(model, data,device, show=False):
def visualized(model, data, device, data_set, data_name, source_name, target_name, open_set, show=True):  
    source_feats = [model.get_feats(data[i]['X'], device=device) for i in range(len(data))]
    feats = np.concatenate(source_feats, axis=0)
    y = [data[i]['Y'] for i in range(len(data))]
    labels = np.concatenate(y)
    domain_labels = []
    for i in range(len(data)):
        domain_labels += [i] * data[i]['X'].shape[0]
    domain_labels = np.array(domain_labels)

    if show:
        tsne = TSNE(n_components=2, init="pca", learning_rate='auto')
        feats_tsne = tsne.fit_transform(feats)
        plot_embedding(feats_tsne, labels, d=domain_labels)

    result_name = "SAFAARI_supervised_Integration_Results/{}/{}/".format(data_set, data_name)
    if not os.path.exists(result_name):
        os.makedirs(result_name)

    if open_set:
        output_name_s = "SAFAARI_supervised_Integration_Results/{}/{}/{}_{}_to_{}_embeddings_op_source.csv".format(data_set, data_name, data_name, source_name, target_name)
    else:
        output_name_s = "SAFAARI_supervised_Integration_Results/{}/{}/{}_{}_to_{}_embeddings_source.csv".format(data_set, data_name, data_name, source_name, target_name)
    df_s = pd.DataFrame(source_feats[0])
    df_s.to_csv(output_name_s)

    if open_set:
        output_name_t = "SAFAARI_supervised_Integration_Results/{}/{}/{}_{}_to_{}_embeddings_op_target.csv".format(data_set, data_name, data_name,
                                                                                         source_name, target_name)
    else:
        output_name_t = "SAFAARI_supervised_Integration_Results/{}/{}/{}_{}_to_{}_embeddings_target.csv".format(data_set, data_name, data_name,
                                                                                      source_name, target_name)
    df_t = pd.DataFrame(source_feats[1])
    df_t.to_csv(output_name_t)

    print(f"Source embeddings saved to: {output_name_s}")
    print(f"Target embeddings saved to: {output_name_t}")

#def train(args, data, device):
def train(args, data, device, data_set, data_name, source_name, target_name, open_set):  

    ori_data = copy.deepcopy(data)

    oversample = SMOTE()
    for i in range(len(data)):
        X, y = oversample.fit_resample(data[i]['X'], data[i]['Y'])
        data[i]['X'] = X
        data[i]['Y'] = y

    model = OuterAdapterModel(args=args, num_shared_classes=max(data[0]['Y'])+1, input_dim=data[1]['X'].shape[1]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.moment, weight_decay=args.l2_decay)
    args.batch_size = min(args.batch_size, data[1]['X'].shape[0])
    data_generator = [batch_generator(data[i], args.batch_size) for i in range(len(data))]

    # train the model with epochs: args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        all_sinputs, all_slabels = [], []
        for i in range(len(data)):
            sinputs, slabels = next(data_generator[i])
            sinputs = torch.tensor(sinputs, requires_grad=False).to(device)
            slabels = torch.tensor(slabels, requires_grad=False, dtype=torch.long).to(device)
            all_sinputs.append(sinputs)
            all_slabels.append(slabels)

        optimizer.zero_grad()
        loss = model(all_sinputs, all_slabels)
        loss.backward()
        optimizer.step()
        if np.isnan(loss.item()):
            print("-"*100, "Error!")
        #print("Generating Embeddings...", loss.item())

    print("Generating Embeddings...")
    #visualized(model, ori_data, device)
    visualized(model, ori_data, device, data_set, data_name, source_name, target_name, open_set)  
    del model
    del data, ori_data


def load_data_FACS10X(path, data_name, domain_name, use_unknown):
    if data_name == "mammary":
        all_dicts = ['endothelial cell', 'stromal cell', 'basal cell', 'macrophage']
    elif data_name == "spleen":
        all_dicts = ['T cell', 'B cell', 'dendritic cell']
    elif data_name == "kidney":
        all_dicts = ['endothelial cell', 'kidney tubule cell', 'fenestrated cell', 'fibroblast',
                     'kidney collecting duct cell', 'leukocyte']
    elif data_name == "bladder":
        all_dicts = ['bladder cell', 'mesenchymal cell', 'leukocyte']
    elif data_name == "muscle":
        all_dicts = ['B cell', 'T cell', 'endothelial cell', 'skeletal muscle satellite cell', 'mesenchymal stem cell',
                     'macrophage', 'chondroblast']
    elif data_name == "heart":
        all_dicts = ['endocardial cell', 'fibroblast', 'cardiac muscle cell', 'endothelial cell',
                     'erythrocyte', 'smooth muscle cell']
    elif data_name == "marrow":
        all_dicts = ['granulocyte', 'T cell', 'Fraction A pre-pro B cell', 'monocyte', 'hematopoietic stem cell',
                     'B cell', 'erythrocyte']
    elif data_name == "liver":
        all_dicts = ['endothelial cell', 'hepatocyte', 'unknown']
    df = pd.read_csv('{}/FACS10X/{}/{}_{}__CmnGenes.csv'.format(path, data_name, data_name, domain_name))
    features = df[df.columns[1:]].to_numpy().astype(np.float32)
    raw_labels = df["cell types"].tolist()

    num_samples = np.zeros(len(all_dicts)).astype(int)
    labels = []
    print(set(raw_labels))
    for i in range(len(raw_labels)):
        la = raw_labels[i]
        ind = all_dicts.index(la)
        labels.append(ind)
        num_samples[ind] += 1
    labels = np.array(labels)

    for i in range(len(all_dicts)):
        print("Class {}: {}".format(i, all_dicts[i]))
    print("\n")
    print("Data Summary: {}".format(domain_name))
    for i in range(len(all_dicts)):
        print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))

    data = {}
    if use_unknown:
        data['X'] = features
        data['Y'] = labels
    else:
        data['X'] = features[labels < len(all_dicts) - 1]
        data['Y'] = labels[labels < len(all_dicts) - 1]
    return data


def load_data_muris(path, data_name, domain_name, use_unknown):
    if data_name == "liver":
        all_dicts = ['hepatocyte', 'endothelial cell', 'B cell']
    elif data_name == "mammary":
        all_dicts = ['endothelial cell', 'stromal cell', 'basal cell', 'luminal epithelial cell of mammary gland']
    elif data_name == "spleen":
        all_dicts = ['T cell', 'B cell', 'myeloid cell']
    elif data_name == "kidney":
        all_dicts = ['endothelial cell', 'kidney tubule cell', 'fenestrated cell', 'fibroblast',
                     'kidney collecting duct cell', 'leukocyte']
    elif data_name == "bladder":
        all_dicts = ['bladder cell', 'mesenchymal cell', 'basal cell of urothelium']
    elif data_name == "muscle":
        all_dicts = ['B cell', 'T cell', 'endothelial cell', 'skeletal muscle satellite cell', 'mesenchymal stem cell',
                     'macrophage', 'skeletal muscle satellite stem cell']
    elif data_name == "heart":
        all_dicts = ['smooth muscle cell', 'endocardial cell', 'fibroblast', 'cardiac muscle cell', 'endothelial cell',
                     'erythrocyte', 'epicardial adipocyte']
    elif data_name == "marrow":
        all_dicts = ['granulocyte', 'T cell', 'Fraction A pre-pro B cell', 'monocyte', 'hematopoietic stem cell',
                     'B cell', 'natural killer cell']
    df = pd.read_csv('{}/FACS10X/{}/{}_{}_CmnGenes.csv'.format(path, data_name, data_name, domain_name))
    features = df[df.columns[1:]].to_numpy().astype(np.float32)
    raw_labels = df["cell types"].tolist()

    num_samples = np.zeros(len(all_dicts)).astype(int)
    labels = []
    print(set(raw_labels))
    for i in range(len(raw_labels)):
        la = raw_labels[i]
        ind = all_dicts.index(la)
        labels.append(ind)
        num_samples[ind] += 1
    labels = np.array(labels)

    for i in range(len(all_dicts)):
        print("Class {}: {}".format(i, all_dicts[i]))
    print("\n")
    print("Data Summary: {}".format(domain_name))
    for i in range(len(all_dicts)):
        print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))

    data = {}
    if use_unknown:
        data['X'] = features
        data['Y'] = labels
    else:
        data['X'] = features[labels < len(all_dicts) - 1]
        data['Y'] = labels[labels < len(all_dicts) - 1]
    return data


def load_data_ovary(path, data_name, domain_name, use_unknown=None):
    all_dicts = ["endothelial cell", "smooth muscle cell", "stromal cell", "pericyte", "leukocyte", "endothelial cell of lymphatic vessel"]
    df = pd.read_csv('{}/{}/ovary_{}.csv'.format(path, data_name, domain_name))
    features = df[df.columns[1:]].to_numpy().astype(np.float32)
    raw_labels = df["Cell_Type"].tolist()

    num_samples = np.zeros(len(all_dicts)).astype(int)
    labels = []
    for i in range(len(raw_labels)):
        la = raw_labels[i]
        ind = all_dicts.index(la)
        labels.append(ind)
        num_samples[ind] += 1
    labels = np.array(labels)
    print("Data Summary: {}".format(domain_name))
    for i in range(len(all_dicts)):
        if use_unknown:
            print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))
        elif i < len(all_dicts) - 1:
            print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))

    data = {}
    if use_unknown:
        data['X'] = features
        data['Y'] = labels
    else:
        data['X'] = features[labels < len(all_dicts) - 1]
        data['Y'] = labels[labels < len(all_dicts) - 1]
    return data

def load_data_PBMC(path, data_name, domain_name, use_unknown=None):
    all_dicts = ['CD8 T', 'other', 'Mono', 'other T', 'NK', 'DC', 'B', 'CD4 T']
    df = pd.read_csv('{}/{}/{}.csv'.format(path, data_name, domain_name))
    features = df[df.columns[1:]].to_numpy().astype(np.float32)
    raw_labels = df["cell_type"].tolist()
    print(set(raw_labels))

    num_samples = np.zeros(len(all_dicts)).astype(int)
    labels = []
    for i in range(len(raw_labels)):
        la = raw_labels[i]
        ind = all_dicts.index(la)
        labels.append(ind)
        num_samples[ind] += 1
    labels = np.array(labels)
    print("Data Summary: {}".format(domain_name))
    for i in range(len(all_dicts)):
        if use_unknown:
            print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))
        elif i < len(all_dicts) - 1:
            print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))

    data = {}
    data['X'] = features
    data['Y'] = labels
    return data

def load_data_SEURAT_PBMC(path, data_name, domain_name, use_unknown):
    all_dicts = ['MAIT', 'NK', 'Naive B', 'CD8 Naive', 'CD4 TCM', 'gdT', 'CD16 Mono', 'CD14 Mono', 'CD8 TEM_1', 'Treg',
                 'Memory B', 'pDC', 'CD4 TEM', 'CD4 Naive', 'Intermediate B', 'cDC', 'CD8 TEM_2', 'Plasma', 'HSPC']
    df = pd.read_csv('{}/{}/{}_data_with_cell_type.csv'.format(path, data_name, domain_name))
    features = df[df.columns[1:]].to_numpy().astype(np.float32)
    raw_labels = df["cell_type"].tolist()
    print(set(raw_labels))

    num_samples = np.zeros(len(all_dicts)).astype(int)
    labels = []
    for i in range(len(raw_labels)):
        la = raw_labels[i]
        ind = all_dicts.index(la)
        labels.append(ind)
        num_samples[ind] += 1
    labels = np.array(labels)
    print("Data Summary: {}".format(domain_name))
    for i in range(len(all_dicts)):
        print("Class {}: {}, Number of samples: {}".format(i, all_dicts[i], num_samples[i]))

    data = {}
    data['X'] = features
    data['Y'] = labels
    return data


def main (): #if __name__ == '__main__':
    args = get_args()  # Get command-line arguments
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)

    data_set = "FACS10X"
    source_name = "FACS"
    target_name = "10X"
    open_set = True
    for data_name in ["bladder", "heart", "kidney", "mammary", "marrow", "muscle", "spleen"]: #, "liver"]:
        print("*"*100, data_name)
    # data_set = "Muris"
    # source_name = "10X"
    # target_name = "FACS"
    # open_set = True
    # for data_name in ["bladder", "heart", "kidney", "liver", "mammary", "marrow", "muscle", "spleen"]:
    #     print("*"*100, data_name)
        data = []
        if data_set == "FACS10X":
            ds = load_data_FACS10X(path="data", data_name=data_name, domain_name=source_name, use_unknown=False)
            dt = load_data_FACS10X(path="data", data_name=data_name, domain_name=target_name, use_unknown=open_set)
        elif data_set == "Muris":
            ds = load_data_muris(path="data", data_name=data_name, domain_name=source_name, use_unknown=False)
            dt = load_data_muris(path="data", data_name=data_name, domain_name=target_name, use_unknown=open_set)
        else:
            print("Unknown data!")
        data.append(ds)
        data.append(dt)
        #train(args, data, device)
        train(args, data, device, data_set, data_name, source_name, target_name, open_set)  # Pass all arguments

    # data_set = "Ovary"
    # data_name = "Ovary"
    # source_name = "RNA"
    # target_name = "ATAC"
    # open_set = False
    # ds = load_data_ovary(path="data", data_name=data_name, domain_name=source_name, use_unknown=False)
    # dt = load_data_ovary(path="data", data_name=data_name, domain_name=target_name, use_unknown=open_set)
    # data = []
    # data.append(ds)
    # data.append(dt)
    # train(args, data, device)

    # data_set = "PBMC_ADT_SCT"
    # data_name = "PBMC_ADT_SCT"
    # source_name = "ADT"
    # target_name = "SCT"
    # open_set = False
    # ds = load_data_PBMC(path="data", data_name=data_name, domain_name=source_name, use_unknown=False)
    # dt = load_data_PBMC(path="data", data_name=data_name, domain_name=target_name, use_unknown=open_set)
    # data = []
    # data.append(ds)
    # data.append(dt)
    # train(args, data, device)

    # data_set = "SEURAT_PBMC_RNA_ATAC"
    # data_name = "SEURAT_PBMC_RNA_ATAC"
    # source_name = "rna"
    # target_name = "atac"
    # open_set = False
    # ds = load_data_SEURAT_PBMC(path="data", data_name=data_name, domain_name=source_name, use_unknown=False)
    # dt = load_data_SEURAT_PBMC(path="data", data_name=data_name, domain_name=target_name, use_unknown=open_set)
    # data = []
    # data.append(ds)
    # data.append(dt)
    # train(args, data, device)


if __name__ == "__main__":
    main()
