import torch

def crossEntropy(logits, label_ids):
    loss = torch.tensor(0.0).cuda()
    logits = torch.sigmoid(logits).view(-1)
    for i, label_item in enumerate(label_ids):
        # import pdb; pdb.set_trace()
        if(label_item == 1):
            if(logits[i] != 0):
                loss += torch.log(logits[i])
        else:
            if(logits[i] != 1):
                loss += torch.log(1 - logits[i])
    loss = 0- loss
    return loss


def hingeLoss(logits, label_ids):
    loss = torch.tensor(0.0).cuda()
    scores = torch.sigmoid(logits).view(-1, 2)
    pos_score = scores.view(-1, 2)[:, 0].view(-1)
    neg_score = scores.view(-1, 2)[:, 1].view(-1)
    margin_loss = torch.nn.functional.relu(neg_score + 0.5 - pos_score)
    loss = torch.mean(margin_loss)
    return loss