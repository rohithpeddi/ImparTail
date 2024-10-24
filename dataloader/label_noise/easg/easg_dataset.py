from dataloader.base_easg_dataset import BaseEASGData


class LabelNoiseEASG(BaseEASGData):

    def __init__(self, conf):
        super().__init__(conf)

    def __getitem__(self, idx):
        return self.graphs[idx]