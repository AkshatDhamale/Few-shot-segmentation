import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from . import image_transforms as myit
from .dataset_specifics import *


class TestDataset(Dataset):

    def __init__(self, args):

        # reading the paths
        if args.dataset == 'BRATS':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'BraTS_T2_normalized/image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.image_dirs = self.image_dirs[:len(self.image_dirs)//60]

        # remove test fold!
        # self.FOLD = get_folds(args.dataset)
        # self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args.fold]]

        # split into support/query
        randomize = True
        self.support_dir = []
        if (randomize):            
            if (args.n_shot > 1):
                support_idxs = np.random.randint(len(self.image_dirs), size=args.n_shot)
                self.support_dir.append(self.image_dirs[x] for x in support_idxs)
            else:
                support_idx = np.random.randint(len(self.image_dirs))
                self.support_dir = self.image_dir[support_idx]
        else:
            if (args.n_shot > 1):
                idxs = [0,1]
                self.support_dir.append(self.image_dirs[x] for x in idxs)
            else:
                support_idx = 0
                self.support_dir = self.image_dir[support_idx]
                
        self.support_dir = self.image_dirs[0] # - 1s
        self.image_dirs = self.image_dirs # :-1  # remove support 
        self.label = None 

        # evaluation protocol
        self.EP1 = args.EP1

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

        full_brain_img = img.copy()
        full_brain_img = 1 * (full_brain_img > 0)

        img = (img - img.mean()) / img.std()
        # unbinded_image = img.copy()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        # lbl[lbl == 200] = 1
        # lbl[lbl == 500] = 2
        # lbl[lbl == 600] = 3
        # lbl = 1 * (lbl == self.label)
        lbl = 1 * (lbl > 0)

        full_brain_slices = full_brain_img - lbl

        # new_lbl = (1 * (img > 0)) - lbl

        sample = {'id': img_path}

        # Evaluation protocol 1.
        if self.EP1:
            idx = lbl.sum(axis=(1, 2)) > 0
            sample['image'] = torch.from_numpy(img[idx])
            sample['label'] = torch.from_numpy(lbl[idx])

        # Evaluation protocol 2 (default).
        else:
            # sample['unbinded_image'] = torch.from_numpy(unbinded_image)
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)
            # sample['label'] = torch.from_numpy(new_lbl)

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

        full_brain_img = img.copy()
        full_brain_img = 1 * (full_brain_img > 0)

        img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        # lbl[lbl == 200] = 1
        # lbl[lbl == 500] = 2
        # lbl[lbl == 600] = 3
        # lbl = 1 * (lbl == label)
        lbl = 1 * (lbl > 0)

        full_brain_slices = full_brain_img - lbl

        sample = {}
        if all_slices:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)
            # sample['label'] = torch.from_numpy(new_lbl)
        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = full_brain_slices.sum(axis=(1, 2)) > 0
            # idx = new_lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = torch.from_numpy(img[idx][idx_])
            sample['label'] = torch.from_numpy(lbl[idx][idx_])
            # sample['label'] = torch.from_numpy(new_lbl[idx][idx_])

        return sample


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        self.max_iter = args.max_iterations
        self.read = True  # read images before get_item
        self.train_sampling = 'neighbors'
        self.min_size = 200

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args.dataset == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.image_dirs = self.image_dirs[:len(self.image_dirs)//20]

        self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/label*'))
        # self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, 'supervoxels_' + str(args.n_sv), 'super*'))
        self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.sprvxl_dirs = self.sprvxl_dirs[:len(self.sprvxl_dirs)//20]

        # remove test fold!
        # self.FOLD = get_folds(args.dataset)
        # self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args.fold]]
        # self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args.fold]]

        # read images
        if self.read:
            self.images = {}
            self.sprvxls = {}
            for image_dir, sprvxl_dir in zip(self.image_dirs, self.sprvxl_dirs):
                self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

    def __len__(self):
        # return len(self.image_dirs)
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) > 4:
            n_shot = img.shape[1]
            for shot in range(n_shot):
                cat = np.concatenate((img[0, shot], mask[:, shot])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[0, shot] = cat[:3, :, :]
                mask[:, shot] = np.rint(cat[3:, :, :])

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask[q][None])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[q] = cat[:3, :, :]
                mask[q] = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def __getitem__(self, idx):

        # sample patient idx
        pat_idx = random.choice(range(len(self.image_dirs)))
        # pat_idx = idx

        if self.read:
            # get image/supervoxel volume from dictionary
            img = self.images[self.image_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:
            # read image/supervoxel volume into memory
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))

        # Added
        full_brain_img = img.copy()
        full_brain_img = 1 * (full_brain_img > 0)

        # normalize
        img = (img - img.mean()) / img.std()

        # sample class(es) (supervoxel)
        unique = list(np.unique(sprvxl))
        unique.remove(0)

        size = 0
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1
            while n_slices < ((self.n_shot * self.n_way) + self.n_query):
                # cls_idx = random.choice(unique)

                # extract slices containing the sampled class
                sli_idx = np.sum(sprvxl > 0, axis=(1, 2)) > 0
                # sli_idx = np.sum(sprvxl == cls_idx, axis=(1, 2)) > 0
                n_slices = np.sum(sli_idx)

            img_slices = img[sli_idx]
            sprvxl_slices = 1 * (sprvxl[sli_idx] > 0)

            #Added  
            full_brain_slices = full_brain_img[sli_idx]          

            # sample support and query slices
            i = random.choice(
                np.arange(n_slices - ((self.n_shot * self.n_way) + self.n_query) + 1))  # successive slices
            sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)

            size = np.sum(sprvxl_slices[sample[0]])

        # invert order
        if np.random.random(1) > 0.5:
            sample = sample[::-1]  # successive slices (inverted)

        # sup_lbl = sprvxl_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        # qry_lbl = sprvxl_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W

        # Added        
        full_brain_slices = full_brain_slices - sprvxl_slices

        sup_lbl = full_brain_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        qry_lbl = full_brain_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W

        sup_img = img_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)

        # gamma transform
        # if np.random.random(1) > 0.5:
        #     qry_img = self.gamma_tansform(qry_img)
        # else:
        #     sup_img = self.gamma_tansform(sup_img)

        # # geom transform
        # if np.random.random(1) > 0.5:
        #     qry_img, qry_lbl = self.geom_transform(qry_img, qry_lbl)
        # else:
        #     sup_img, sup_lbl = self.geom_transform(sup_img, sup_lbl)

        sample = {'support_images': sup_img,
                  'support_fg_labels': sup_lbl,
                  'query_images': qry_img,
                  'query_labels': qry_lbl}

        return sample
