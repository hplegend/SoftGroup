from .custom import CustomDataset
import numpy as np
import torch

class STPLS3DDataset(CustomDataset):

    CLASSES = ('stem', 'leaf')

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        # instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        instance_cls = [x if x != -100 else x for x in instance_cls] # x 到底要不要从0开始，test的结果是从1开始的。
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    # 覆盖
    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, dummy_sem_label, dummy_inst_label
