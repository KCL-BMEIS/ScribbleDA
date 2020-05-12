import torch
from torch import nn

class DC_CE(nn.Module):
    def __init__(self,nb_classes):
        super(DC_CE, self).__init__()

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod 
    def onehot(gt,shape):
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
        return y_onehot

    def reshape(self,output, target):
        batch_size = output.shape[0]

        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        target = target.permute(0,2,3,4,1)
        output = output.permute(0,2,3,4,1)
        print(target.shape,output.shape)
        return output, target


    def dice(self,output, target):
        output = self.softmax(output)
        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        sum_axis = list(range(2,len(target.shape)))

        s = (10e-20)
        intersect = torch.sum(output * target,sum_axis)
        dice = (2 * intersect) / (torch.sum(output,sum_axis) + torch.sum(target,sum_axis) + s)
        #dice shape is (batch_size, nb_classes)
        return 1.0 - dice.mean()  


    def forward(self, output, target):
        dc_loss = self.dice(output, target)

        output = output.permute(0,2,3,4,1).contiguous().view(-1,self.nb_classes)
        target = target.view(-1,).long().cuda()
        ce_loss = self.ce(output, target)
        result = ce_loss + dc_loss
        return result

