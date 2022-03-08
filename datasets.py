import torch
import numpy as np
import augmentations

class SensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transforms=None, input_repeat=1):
        self.x = x
        self.y = y
        self.input_repeat = input_repeat
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_out = self.x[[idx]] # for data augmentation, the data is transformed to (1, ch, winsize).
        y_out = self.y[[idx]]

        if self.input_repeat > 1:
            x_out = np.tile(x_out, (self.input_repeat, 1))

        if self.transforms is not None:
            for t in self.transforms:
                x_out, y_out = t(x_out, y_out)

        return x_out[0], y_out[0]   # Return to the original shape

class SensorMaskedDataset(SensorDataset):
    def __init__(self, x, y, mask, transforms=None, input_repeat=1):
        super().__init__(x, y, transforms=transforms, input_repeat=input_repeat)
        self.mask = mask

    def __getitem__(self, idx):
        x_out, y_out = super().__getitem__(idx)
        x_out = x_out[np.newaxis, :, :]
        mask_out = self.mask[idx]

        outputs = []
        for i, mask in enumerate(mask_out.T):
            outputs.append((x_out.T * mask).T)
        x_out = np.concatenate(outputs, axis=1)  # c

        return x_out[0], y_out

class SensorAugmentedDataset(SensorDataset):
    def __init__(self, x, y, transforms=None, input_repeat=1, ensembles=1, additional_transforms=None, type="fixed"):
        super().__init__(x, y, transforms=transforms, input_repeat=input_repeat)
        self.additional_transforms = additional_transforms
        self.ensembles = ensembles
        self.type=type

    def __getitem__(self, idx):
        x_out, y_out = super().__getitem__(idx)
        x_out = x_out[np.newaxis, :, :]

        if self.additional_transforms is not None:
            x_aug = []
            trans_idx = np.arange(len(self.additional_transforms))
            if self.type == "random":
                np.random.shuffle(trans_idx)
            for i in trans_idx:
                xa, _ = self.additional_transforms[i](x_out, y_out)
                x_aug.append(xa)
            x_out = np.concatenate(x_aug, axis=1)
        else:
            x_out = np.tile(x_out, (self.ensembles, 1))

        return x_out[0].astype(np.float32), y_out

class SensorAugmentedMaskedDataset(SensorAugmentedDataset):
    def __init__(self, x, y, mask, transforms=None, input_repeat=1, ensembles=1, additional_transforms=None, type="fixed"):
        super().__init__(x, y, transforms=transforms, input_repeat=input_repeat, ensembles=ensembles, additional_transforms=additional_transforms, type=type)
        self.mask = mask

    def __getitem__(self, idx):
        x_out, y_out = super().__getitem__(idx)
        x_out = x_out[np.newaxis, :, :]
        mask_out = self.mask[idx]

        outputs = []
        for i, mask in enumerate(mask_out.T):
            outputs.append((x_out.T * mask).T)
        x_out = np.concatenate(outputs, axis=1)  # c

        return x_out[0], y_out

def create_datasets(train_, val_, test_, input_repeat=1, transforms=[]):
    dl_all = []
    dl_all.append(SensorDataset(train_[0].astype(np.float32), train_[1], input_repeat=input_repeat, transforms=transforms))
    dl_all.append(SensorDataset(val_[0].astype(np.float32), val_[1], input_repeat=input_repeat))
    dl_all.append(SensorDataset(test_[0].astype(np.float32), test_[1], input_repeat=input_repeat))
    return dl_all

def create_masks(target, ensembles):
    tlist = np.sort(np.unique(target))
    tlen = len(tlist)
    folds = np.tile(np.arange(ensembles), int(np.ceil(tlen/ensembles)))[:tlen]
    masks = []
    for i in range(ensembles):
        tgt = tlist[np.where(folds != i)]
        flag = [t in tgt for t in target]
        masks.append(flag*1)
    masks = np.stack(masks).T*1
    return masks

def create_masked_datasets(train_, val_, test_, ensembles, input_repeat=1, transforms=[], type='random'):
    bs = len(train_[0])
    if type == 'random':
        rlist = np.tile(np.arange(ensembles), int(bs/ensembles)+1)[:bs]
        masks = create_masks(rlist, ensembles)
    elif type == 'all':
        masks = np.ones([bs, ensembles]).astype(np.int8)
    dl_mask = []
    dl_mask.append(SensorMaskedDataset(train_[0].astype(np.float32), train_[1], masks, input_repeat=input_repeat, transforms=transforms))
    dl_mask.append(SensorMaskedDataset(val_[0].astype(np.float32), val_[1], np.ones([len(val_[1]), ensembles]).astype(np.int8), input_repeat=input_repeat))
    dl_mask.append(SensorMaskedDataset(test_[0].astype(np.float32), test_[1], np.ones([len(test_[1]), ensembles]).astype(np.int8), input_repeat=input_repeat))
    return dl_mask

def create_augmented_datasets(train_, val_, test_, input_repeat=1, transforms=[], type='fixed'):
    at = [augmentations.jitter(m=3), augmentations.scale(m=4), augmentations.vershift(m=2), augmentations.jitter(m=5)]
    dl_aug = []
    dl_aug.append(SensorAugmentedDataset(train_[0].astype(np.float32), train_[1], ensembles=4, input_repeat=input_repeat, transforms=transforms, additional_transforms=at, type=type))
    dl_aug.append(SensorAugmentedDataset(val_[0].astype(np.float32), val_[1], ensembles=4, input_repeat=input_repeat))
    dl_aug.append(SensorAugmentedDataset(test_[0].astype(np.float32), test_[1], ensembles=4, input_repeat=input_repeat))
    return dl_aug

def create_augmented_masked_datasets(train_, val_, test_, ensembles, input_repeat=1, transforms=[], mtype="random", atype='fixed'):
    bs = len(train_[0])
    if mtype == 'random':
        rlist = np.tile(np.arange(ensembles), int(bs/ensembles)+1)[:bs]
        masks = create_masks(rlist, ensembles)
    elif mtype == 'all':
        masks = np.ones([bs, ensembles]).astype(np.int8)
    at = [augmentations.jitter(m=3), augmentations.scale(m=4), augmentations.vershift(m=2), augmentations.jitter(m=5)]
    dl_aug = []
    dl_aug.append(SensorAugmentedMaskedDataset(train_[0].astype(np.float32), train_[1], masks, ensembles=4, input_repeat=input_repeat, transforms=transforms, additional_transforms=at, type=atype))
    dl_aug.append(SensorAugmentedMaskedDataset(val_[0].astype(np.float32), val_[1], np.ones([len(val_[1]), ensembles]).astype(np.int8), ensembles=4, input_repeat=input_repeat))
    dl_aug.append(SensorAugmentedMaskedDataset(test_[0].astype(np.float32), test_[1], np.ones([len(test_[1]), ensembles]).astype(np.int8), ensembles=4, input_repeat=input_repeat))
    return dl_aug