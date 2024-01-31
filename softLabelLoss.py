import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class softLabelLoss(nn.Module):
    def __init__(self, args, mixup_fn) -> None:
        super().__init__()
        lossfn = args.lossfn
        self.nb_class = args.nb_classes
        # self.use_softlabel = args.use_softlabel
        # self.soft_ratio = args.soft_label_ratio
        # self.label_ratio = args.label_ratio
        # self.soft_type = args.soft_type
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            if lossfn == 'BCE':
                self.criterion = nn.BCEWithLogitsLoss()
            elif lossfn == 'CE':
                self.criterion = nn.CrossEntropyLoss()
            elif lossfn == 'L1':
                self.criterion = nn.L1Loss()
            elif lossfn == 'MSE':
                self.criterion = nn.MSELoss()

    
    def partial_onehot(self, target):
        if self.use_softlabel:
            onehot = torch.zeros(len(target), self.nb_class).to(target.device)
            for i, t in enumerate(target):
                if t == 0: # amb_neg
                    onehot[i][0] = self.soft_ratio
                    onehot[i][1] = 1-self.soft_ratio
                # elif t == 1: # amb_pos
                #     onehot[i][0] = 1-self.soft_ratio
                #     onehot[i][1] = self.soft_ratio
                elif t == 1: # amb_pos
                    onehot[i][0] = 1-self.label_ratio
                    onehot[i][1] = self.label_ratio
                elif t == 2: # neg
                    onehot[i][0] = self.label_ratio
                    onehot[i][1] = 1-self.label_ratio
                elif t == 3: # pos
                    onehot[i][0] = 1-self.label_ratio
                    onehot[i][1] = self.label_ratio

                
        else:
            onehot = torch.zeros(len(target), 4).to(target.device)
            for i, t in enumerate(target):
                ## 원래 꺼
                # if self.soft_type == 0:
                #     if t == 0: # amb_neg
                #         onehot[i][0] = self.label_ratio
                #         onehot[i][2] = self.soft_ratio
                #     elif t == 1: # amb_pos
                #         onehot[i][1] = self.label_ratio
                #         onehot[i][3] = self.soft_ratio
                #     elif t == 2: # neg
                #         onehot[i][2] = self.label_ratio
                #         onehot[i][0] = self.soft_ratio
                #     elif t == 3: # pos
                #         onehot[i][3] = self.label_ratio
                #         onehot[i][1] = self.soft_ratio

                # type == 1 : amb_neg + amb_pos = 1
                if self.soft_type == 1:
                    if t == 0: # amb_neg
                        onehot[i][0] = self.soft_ratio
                        onehot[i][1] = 1-self.soft_ratio
                    elif t == 1: # amb_pos
                        onehot[i][1] = self.soft_ratio
                        onehot[i][0] = 1-self.soft_ratio
                    elif t == 2: # neg
                        onehot[i][2] = self.label_ratio
                        onehot[i][0] = 1-self.label_ratio
                    elif t == 3: # pos
                        onehot[i][3] = self.label_ratio
                        onehot[i][1] = 1-self.label_ratio

                    # 이건 neg+amb_neg=1 이게 원래꺼임 (230709. 성능 더 good)
                    # elif t == 2: # neg
                    #     onehot[i][2] = self.label_ratio
                    #     onehot[i][0] = 1-self.label_ratio
                    # elif t == 3: # pos
                    #     onehot[i][3] = self.label_ratio
                    #     onehot[i][1] = 1-self.label_ratio

                    # 이건 neg+pos=1
                    # elif t == 2: # neg
                    #     onehot[i][2] = self.label_ratio
                    #     onehot[i][3] = 1-self.label_ratio
                    # elif t == 3: # pos
                    #     onehot[i][3] = self.label_ratio
                    #     onehot[i][2] = 1-self.label_ratio

                # type == 2 : amb_neg + neg = 1
                elif self.soft_type == 2:
                    if t == 0: # amb_neg
                        onehot[i][0] = self.soft_ratio
                        onehot[i][2] = 1-self.soft_ratio
                    elif t == 1: # amb_pos
                        onehot[i][1] = self.soft_ratio
                        onehot[i][3] = 1-self.soft_ratio
                    elif t == 2: # neg
                        onehot[i][2] = self.label_ratio
                        onehot[i][0] = 1-self.label_ratio
                    elif t == 3: # pos
                        onehot[i][3] = self.label_ratio
                        onehot[i][1] = 1-self.label_ratio            
                
        return onehot


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # if self.nb_class==2 or self.nb_class==4:
        #     target = self.partial_onehot(target)
        # else:
        #     target = torch.nn.functional.one_hot(target, num_classes=self.nb_class).float()
        target = torch.nn.functional.one_hot(target, num_classes=self.nb_class).float()
        loss = self.criterion(input, target)
        return loss