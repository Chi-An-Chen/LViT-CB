import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)).to(self.device)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  # 計算 cos(θ)
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))  # 反餘弦
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * (one_hot * torch.cos(theta + self.m) + (1.0 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.01):
        super(CenterLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.centers  = nn.Parameter(torch.randn(num_classes, feat_dim)).to(self.device)

    def forward(self, features, labels):
        batch_size    = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = ((features - centers_batch) ** 2).sum() / (2.0 * batch_size)
        return self.lambda_c * loss

class CombineLoss(nn.Module):
    def __init__(self, num_classes, dim_in, arcface_weight=0.5, center_weight=0.01):
        super(CombineLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.arcface_loss = ArcFaceLoss(in_features=dim_in, out_features=num_classes).to(self.device)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=dim_in, lambda_c=center_weight).to(self.device)
        self.arcface_weight = arcface_weight
        self.center_weight = center_weight

    def forward(self, output, labels, features):
        loss_ce = self.cross_entropy_loss(output, labels)
        loss_arcface = self.arcface_loss(features, labels)
        loss_center = self.center_loss(features, labels)
        return loss_ce + self.arcface_weight * loss_arcface + self.center_weight * loss_center
