import torch
from scipy.stats import kendalltau, pearsonr, spearmanr


def compute_all_corr(x, y):
    kendall = torch.tensor(kendalltau(x, y)[0], dtype=torch.float32)
    pearson = torch.tensor(pearsonr(x, y)[0], dtype=torch.float32)
    spearman = torch.tensor(spearmanr(x, y)[0], dtype=torch.float32)
    return kendall, pearson, spearman

