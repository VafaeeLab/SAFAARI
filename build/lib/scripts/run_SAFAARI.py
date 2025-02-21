import argparse
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import batch_generator, plot_embedding, set_random_seed, one_hot
from scripts.utils import batch_generator, plot_embedding, set_random_seed, one_hot

import copy
import pickle
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from models.SAFAARI import SAFAARIModel


# Command setting
parser = argparse.ArgumentParser(description='Open Set Domain Adaptation')
parser.add_argument('--open_set', type=lambda x: x.lower() == 'true', default=True, help='Enable open-set mode (True/False)')
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
args = parser.parse_args()


def visualized(model, data,  device,source_name, target_name,data_name , show=False):

    
    source_feats = [model.get_feats(data[i]['X'], device=device) for i in range(len(data))]
     # Split into source and target embeddings
    source_embeddings = source_feats[0]  # First dataset is source
    target_embeddings = source_feats[1]  # Second dataset is target

    # Convert to Pandas DataFrame
    source_df = pd.DataFrame(source_embeddings)
    target_df = pd.DataFrame(target_embeddings)

    # Ensure the directory exists before saving
    save_path = f"data/results/{data_name}/"
    os.makedirs(save_path, exist_ok=True)

    # Save CSV files separately
    source_csv_path = os.path.join(save_path, f"{source_name}_embeddings.csv")
    target_csv_path = os.path.join(save_path, f"{target_name}_embeddings.csv")

    source_df.to_csv(source_csv_path, index=False)
    target_df.to_csv(target_csv_path, index=False)

    print(f" Source embeddings saved to: {source_csv_path}")
    print(f"Target embeddings saved to: {target_csv_path}")

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

    # if open_set:
    #     with open('data/FACS10X/{}/{}_to_{}_emb_op'.format(data_name, source_name, target_name), "wb") as pkl_file:
    #         pickle.dump(source_feats, pkl_file)
    # else:
    #     with open('data/FACS10X/{}/{}_to_{}_emb'.format(data_name, source_name, target_name), "wb") as pkl_file:
    #         pickle.dump(source_feats, pkl_file)

    

#def train(args, src_data, tgt_data, tgt_test_data, device, data_name,open_set ):
def train(args, src_data, tgt_data, tgt_test_data, device, data_name, source_name, target_name, open_set):

    ori_source_data = copy.deepcopy(src_data)
    ori_target_data = copy.deepcopy(tgt_data)
    num_classes = max(tgt_data['Y']) + 1  # total number of classes
    oversample = SMOTE()
    for i in range(len(src_data)):
        X, y = oversample.fit_resample(src_data[i]['X'], src_data[i]['Y'])
        src_data[i]['X'] = X
        src_data[i]['Y'] = y

    model = SAFAARIModel(args=args, num_classes=num_classes, input_dim=tgt_data['X'].shape[1]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.moment, weight_decay=args.l2_decay)
    args.batch_size = min(args.batch_size, tgt_data['X'].shape[0])
    src_generator = [batch_generator(src_data[i], args.batch_size) for i in range(len(src_data))]
    tgt_generator = batch_generator(tgt_data, args.batch_size)

    # train the model with epochs: args.epochs
    for s in range(len(src_data)):
        for epoch in range(1, args.epochs + 1):
            model.train()
            all_sinputs, all_slabels = [], []
            for i in range(s+1):
                sinputs, slabels = next(src_generator[i])
                sinputs = torch.tensor(sinputs, requires_grad=False).to(device)
                slabels = torch.tensor(slabels, requires_grad=False, dtype=torch.long).to(device)
                all_sinputs.append(sinputs)
                all_slabels.append(slabels)
            all_sinputs = torch.cat(all_sinputs, dim=0)
            all_slabels = torch.cat(all_slabels, dim=0)
            if s == len(src_data) - 1:
                adapt = True
                tinputs, t_cluster_idx = next(tgt_generator)
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
            else:
                adapt = False
                tinputs, t_cluster_idx = next(src_generator[s+1])
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                t_cluster_idx = torch.tensor(t_cluster_idx, requires_grad=False, dtype=torch.long).to(device)

            optimizer.zero_grad()
            loss = model(all_sinputs, all_slabels, tinputs, t_cluster_idx, epoch, adapt=adapt, open_set=open_set, data_name=data_name)
            loss.backward()
            optimizer.step()
            # if epoch % args.log_interval == 0:
            #     results = test(model, tgt_test_data, num_classes, device=device, print_results=True)
        vis_data = ori_source_data[:s+1]
        if s == len(src_data) - 1:
            vis_data += [ori_target_data]
        else:
            vis_data += [ori_source_data[s+1]]

        visualized(model, vis_data,  device, source_name, target_name, data_name, show=False)


    # results = test(model, tgt_test_data, num_classes, device=device, open_set , print_results=True)
    results = test(model, tgt_test_data, num_classes, device=device, open_set=open_set, 
               data_name=data_name, source_name=source_name, target_name=target_name, print_results=True)

    del model
    del src_data, tgt_data, tgt_test_data
    return results


#def test(model, tgt_test_data, num_classes, device, open_set, print_results=False):
def test(model, tgt_test_data, num_classes, device, open_set, data_name, source_name, target_name, print_results=False):
    print(f"Inside test(): open_set = {open_set}")  # Debugging print
    model.eval()
    per_class_num = np.zeros((num_classes))
    per_class_correct = np.zeros((num_classes))

    img_t = torch.tensor(tgt_test_data['X'], requires_grad=False).to(device)
    label_t = torch.tensor(tgt_test_data['Y'], requires_grad=False, dtype=torch.long).to(device)
    out_t = model.inference(img_t)
    pred = out_t.data.max(1)[1]

    # save results
    gt_and_pred = np.array([label_t.data.cpu().numpy(), pred.data.cpu().numpy()]).T
    column_names = ['True Label', 'Predicted Label']
    df = pd.DataFrame(gt_and_pred, columns=column_names)
    if open_set:
        df.to_csv("data/results/{}/{}_to_{}_labels_op.csv".format(data_name, source_name, target_name))
    else:
        df.to_csv("data/results/{}/{}_to_{}_labels.csv".format(data_name, source_name, target_name))

    for t in range(num_classes):
        t_ind = np.where(label_t.data.cpu().numpy() == t)
        correct_ind = np.where(pred[t_ind[0]] == t)
        per_class_correct[t] += float(len(correct_ind[0]))
        per_class_num[t] += float(len(t_ind[0]))

    label_t_one_hot = one_hot(label_t, num_classes)
    one_hot_pred_ = one_hot(pred, num_classes)
    cm = confusion_matrix(label_t_one_hot.argmax(axis=1), one_hot_pred_.argmax(axis=1))
    recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
    precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    per_class_acc = per_class_correct / per_class_num  # It outputs the accuracy for each class
    per_class_acc = per_class_acc[per_class_num > 1e-5]
    if print_results:
        print("per_class_acc", per_class_acc)
        print("per_class_num:", per_class_num)
        print("per_class_correct:", per_class_correct)
        print("Confusion Matrix \n", cm)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)
        print('IT is from test:', per_class_acc.mean(), per_class_acc[:-1].mean(), per_class_acc[-1])
    return per_class_acc.mean(), per_class_acc[:-1].mean(), per_class_acc[-1]


def load_data(path, data_name, domain_name, use_unknown):
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


#if __name__ == '__main__':
def main():
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)
    #print(f"Running SAFAARI with open_set in main = {args.open_set}")  # Debugging print
    if args.open_set:
        print("SAFAARI is running in Open Set mode")
    else:
        print("SAFAARI is running in Closed Set mode")
 

    source_domain_names = ["FACS"]
    target_domain_names = ["10X"]
    open_set = args.open_set  # Get from user input True#False #True #False
    for data_name in ["bladder"]:#, "heart", "kidney", "mammary", "marrow", "muscle", "spleen"]:
        print("*"*100, data_name)
        for target_name in target_domain_names:
            source_data = []
            for source_name in source_domain_names:
                data = load_data(path="data", data_name=data_name, domain_name=source_name, use_unknown=False)
                source_data.append(data)
            target_data = load_data(path="data", data_name=data_name, domain_name=target_name, use_unknown=open_set)
            target_test_data = copy.deepcopy(target_data)
            train(args, source_data, target_data, target_test_data, device, data_name,source_name, target_name, open_set)

            

if __name__ == "__main__":
    main()