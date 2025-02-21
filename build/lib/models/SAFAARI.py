import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.losses import SupConLoss


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


class SAFAARIModel(nn.Module):
    def __init__(self, args, num_classes, input_dim, bottleneck_width=32, width=168):
        super(SAFAARIModel, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.dim = input_dim
        # Two MLP layers for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.dim, width),
            nn.ReLU(),
            nn.Linear(width, bottleneck_width),
            nn.ReLU(),
        )

        self.classifier_layer = nn.Linear(bottleneck_width, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.discriminator = nn.Linear(bottleneck_width, 2)

    def forward(self, s_inputs, s_outputs, t_inputs, t_pseudo_outputs, epoch=None, adapt=False, alpha=1.0, open_set=None, data_name=None):
        s_feats = self.feature_extractor(s_inputs)
        t_feats = self.feature_extractor(t_inputs)
        s_preds = self.classifier_layer(s_feats)
        class_loss = self.criterion(s_preds, s_outputs)

        num_1 = int(s_feats.shape[0] * 0.7)
        t_preds = F.softmax(self.classifier_layer(t_feats), dim=-1)
        p_prob = torch.sum(t_preds[:, :self.num_classes - 1], 1).view(-1, 1)
        idx = torch.argsort(p_prob.flatten(), descending=True)

        domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_feats, t_feats[idx[:num_1]]], dim=0), alpha))
        domain_labels = np.array([0] * s_feats.shape[0] + [1] * num_1)
        domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=t_inputs.device)
        domain_loss = self.criterion(domain_preds, domain_labels)

        if adapt is False:
            t_preds = self.classifier_layer(t_feats)
            class_loss += self.criterion(t_preds, t_pseudo_outputs)

            criterion = SupConLoss(temperature=0.07)
            feats = torch.cat([s_feats, t_feats], dim=0)
            feats_extend = torch.unsqueeze(feats, dim=1)
            feats_extend = F.normalize(feats_extend, dim=-1)
            cl_loss = criterion(feats_extend, torch.cat([s_outputs, t_pseudo_outputs]))
            loss = class_loss + (domain_loss + cl_loss) * 1
        else:
            criterion = SupConLoss(temperature=0.07)
            feats_extend = torch.unsqueeze(s_feats, dim=1)
            feats_extend = F.normalize(feats_extend, dim=-1)
            cl_loss = criterion(feats_extend, s_outputs)
            loss = class_loss + (cl_loss + domain_loss) * 0.01

            if data_name == "muscle":
                num_candicates = 200
                lamb = 0.005
            else:
                num_candicates = 1
                lamb = 0.01

            if epoch > 0:
                if open_set:
                    with torch.no_grad():
                        feats = self.feature_extractor(copy.deepcopy(t_inputs))
                        preds = self.classifier_layer(feats)
                        pseudo_prob = torch.softmax(preds, dim=-1)
                        max_probs, pseudo_label = torch.max(pseudo_prob, dim=-1)
                        idx = torch.argsort(max_probs.flatten(), descending=False)[:num_candicates]
                        pseudo_label[idx] = self.num_classes - 1
                    t_feats = self.feature_extractor(t_inputs)
                    t_preds = self.classifier_layer(t_feats)
                    loss += lamb * self.criterion(t_preds[pseudo_label==self.num_classes - 1], pseudo_label[pseudo_label==self.num_classes - 1])

                cluster_loss = 0
                pseudo_label = torch.max(self.classifier_layer(t_feats), dim=-1)[1]
                for i in range(self.num_classes-1):
                    s_class_feats = s_feats[s_outputs == i]
                    t_class_feats = t_feats[pseudo_label == i]
                    cluster_loss += torch.cdist(s_class_feats, t_class_feats, p=2).mean() * 2
                    cluster_loss += torch.cdist(s_class_feats, s_class_feats, p=2).mean()
                    cluster_loss += torch.cdist(t_class_feats, t_class_feats, p=2).mean()
                loss += cluster_loss * 0.0001

        return loss

    def one_hot(self, x):
        # represent class label as one-hot vector
        out = torch.zeros(len(x), self.num_classes).to(x.device)
        out[torch.arange(len(x)), x.squeeze()] = 1
        return out

    def inference(self, x):
        # predict the class label for testing example
        x = self.feature_extractor(x).view(x.shape[0], -1)
        return self.classifier_layer(x)

    def get_feats(self, x, device):
        x = torch.tensor(x, requires_grad=False).to(device)
        x = self.feature_extractor(x).view(x.shape[0], -1)
        return x.cpu().detach().numpy()
