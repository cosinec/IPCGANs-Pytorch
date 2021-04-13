import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from utils.network import Conv2d #same padding

'''
class AlexNet_Feature_Extractor(nn.Module):
    #for feature loss
    def __init__(self,pretrainded=False,modelpath=None):
        super(AlexNet_Feature_Extractor, self).__init__()
        assert pretrainded is False or modelpath is not None,"pretrain model need to be specified"
        self.relu=nn.ReLU()
        self.Maxpool=nn.MaxPool2d(3,2)
        self.Conv1=nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.Conv2=nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.Conv3=nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.Conv4=nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.Conv5=nn.Conv2d(256, 256, kernel_size=3, padding=1)

        if pretrainded is True:
            self.load_pretrained_params(modelpath)

    def forward(self, x):
        conv1=self.Conv1(x)
        pool1=self.Maxpool(self.relu(conv1))

        conv2 = self.Conv2(pool1)
        pool2 = self.Maxpool(self.relu(conv2))

        conv3 = self.Conv3(pool2)
        relu3 = self.relu(conv3)

        conv4=self.Conv4(relu3)
        relu4=self.relu(conv4)

        conv5=self.Conv5(relu4)
        return conv5

    def load_pretrained_params(self,path):
        # step1: load pretrained model
        pretrained_dict = torch.load(path)
        # step2: get model state_dict
        model_dict = self.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.load_state_dict(model_dict)
'''

class AgeAlexNet(nn.Module):
    def __init__(self,pretrainded=False,modelpath=None):
        super(AgeAlexNet, self).__init__()
        assert pretrainded is False or modelpath is not None,"pretrain model need to be specified"
        self.features = nn.Sequential(
            Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2,2e-5,0.75),

            Conv2d(96, 256, kernel_size=5, stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2, 2e-5, 0.75),

            Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            Conv2d(384, 384, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),

            Conv2d(384, 256, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.age_classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),
        )
        if pretrainded is True:
            self.load_pretrained_params(modelpath)

        self.Conv3_feature_module=nn.Sequential()
        self.Conv4_feature_module=nn.Sequential()
        self.Conv5_feature_module=nn.Sequential()
        self.Pool5_feature_module=nn.Sequential()
        for x in range(10):
            self.Conv3_feature_module.add_module(str(x), self.features[x])
        for x in range(10,12):
            self.Conv4_feature_module.add_module(str(x),self.features[x])
        for x in range(12,14):
            self.Conv5_feature_module.add_module(str(x),self.features[x])
        for x in range(14,15):
            self.Pool5_feature_module.add_module(str(x),self.features[x])


    def forward(self, x):
        self.conv3_feature=self.Conv3_feature_module(x)
        self.conv4_feature=self.Conv4_feature_module(self.conv3_feature)
        self.conv5_feature=self.Conv5_feature_module(self.conv4_feature)
        pool5_feature=self.Pool5_feature_module(self.conv5_feature)
        self.pool5_feature=pool5_feature
        flattened = pool5_feature.view(pool5_feature.size(0), -1)
        age_logit = self.age_classifier(flattened)
        return age_logit

    def load_pretrained_params(self,path):
        # step1: load pretrained model
        pretrained_dict = torch.load(path)
        # step2: get model state_dict
        model_dict = self.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.load_state_dict(model_dict)


class AgeClassify:
    def __init__(self):
        #step 1:define model
        self.model=AgeAlexNet(pretrainded=False).cuda()
        #step 2:define optimizer 
        #Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
        #它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        #lr (float, optional) ：学习率(默认: 1e-3) 
        #betas (Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
        self.optim=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(0.5, 0.999)) 
        #step 3:define loss
        self.criterion=nn.CrossEntropyLoss().cuda() #交叉熵损失函数

    def train(self,input,label):
        self.model.train()
        output=self.model(input)
        self.loss=self.criterion(output,label)

    def val(self,input):
        self.model.eval()
        output=F.softmax(self.model(input),dim=1).max(1)[1]
        return output

    def save_model(self,dir,filename):
        torch.save(self.model.state_dict(),os.path.join(dir,filename))
if __name__=="__main__":
    one=torch.ones((1,3,227,227))
    model=AgeAlexNet()
