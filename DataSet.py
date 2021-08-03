from torch.utils.data import dataset
import os
from PIL import Image
from torchvision.transforms import transforms
import torch
from torchvision.transforms import functional as F
import numpy as np
import glob
import random
import imageio, cv2
import math


class BaseDataSet(dataset.Dataset):

    def __init__(self, config):
        super(BaseDataSet, self).__init__()
        self.config = config

    def loader(self, path, mode):
    # with open(path, 'rb') as f:
        img = Image.open(path)
        return img.convert(mode)

    def label_to_tensor(self, label):
        if isinstance(label, np.ndarray):
            return torch.from_numpy(label)
        else:
            return (F.to_tensor(label)*255.0).type(torch.long)

    def _transform(self, images, tolabel):
        if 'resize' in self.config:
            old_size, _ = images[0].size
            size = [self.config['resize'], self.config['resize']]
            resize = transforms.Resize(size, Image.NEAREST)
            for i in range(len(images)):
                images[i] = resize(images[i])

        if 'hflip' in self.config and self.config['hflip']:
            flip = random.randint(0, 1)
        else:
            flip = 0

        if flip==1:
            for i in range(len(images)):
                images[i] = F.hflip(images[i])

        for i in range(len(images)):
            if tolabel[i]:
                images[i] = self.label_to_tensor(images[i])
            else:
                images[i] = F.to_tensor(images[i])

        return images


class ReconstructDataSet(BaseDataSet):

    def __init__(self, root, config, list_name="image_list.txt"):
        super(ReconstructDataSet, self).__init__(config)
        self.root = root

        self.folders = glob.glob(os.path.join(root, "*"))
        self.folders.sort()

        self.filelist = []
        self.filelists = []


        for i, folder in enumerate(self.folders):

            with open(os.path.join(folder, list_name)) as f:
                filelist = f.readlines()
                filelist.sort(key=int)
                filelist = [(x.strip(), i) for x in filelist]

                self.filelist += filelist
                self.filelists.append(filelist)

        self.size = self.config['resize']
        self.stage = self.config['phase']

    def __len__(self):
        return len(self.filelist)

    # def get_params(self, img, output_size):
    #     """Get parameters for ``crop`` for a random crop.
    #     Args:
    #         img (PIL Image): Image to be cropped.
    #         output_size (tuple): Expected output size of the crop.
    #     Returns:
    #         tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    #     """
    #     #https://github.com/pytorch/vision/blob/7ae1b8c9c00e2bec1b0c111cd1299415519ef002/torchvision/transforms/transforms.py#L473
    #     w,h = img.size
    #
    #     th, tw = output_size
    #     if w == tw and h == th:
    #         return 0, 0, h, w
    #
    #     i = random.randint(0, h - th)
    #     j = random.randint(0, w - tw)
    #     return i, j, th, tw

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def GetTexture(self, im, IUV):
        U = IUV[:, :, 1]
        V = IUV[:, :, 2]
        Texture = np.zeros((24, 128, 128, 3), dtype=np.uint8)
        for PartInd in range(1, 25):
            tex = Texture[PartInd - 1, :, :, :].squeeze()
            x, y = np.where(IUV[:, :, 0] == PartInd)
            u = U[x, y] // 2
            v = V[x, y] // 2
            tex[u, v] = im[x, y]
            Texture[PartInd - 1] = tex
        TextureIm = np.zeros((128 * 4, 128 * 6, 3), dtype=np.uint8)
        for i in range(len(Texture)):
            x = i // 6 * 128
            y = i % 6 * 128
            TextureIm[x:x + 128, y:y + 128] = Texture[i]
        return TextureIm

    def __getitem__(self, index):

        #random crop parameters

        label = self.filelist[index][1]
        name = self.filelist[index][0]
        folder = self.folders[label]

        if self.stage == 'pretrain' or self.stage == 'train':
            image = self.loader(os.path.join(folder, "image", name+".png"), mode="RGB")
            body = self.loader(os.path.join(folder, "body", name+".png"), mode="L")
            foreground = self.loader(os.path.join(folder, "segmentation", name+".png"), mode="L")

            #Targets
            image_index = random.randrange(0, len(self.filelists[label]))
            image_name = self.filelists[label][image_index][0]
            class_image = self.loader(os.path.join(folder, "image", image_name+".png"), mode="RGB")
            class_foreground = self.loader(os.path.join(folder, "segmentation", image_name+".png"), mode="L")
            class_body = self.loader(os.path.join(folder, "body", image_name+".png"), mode="L")
            IUV = self.loader(os.path.join(folder, "densepose", name+".png"), mode="RGB")

            print (image)
            i, j, h, w = self.get_params(image, scale=(0.08, 1.0), ratio=(1.0, 1.0))
            print (i,j,h,w)
            new_im = F.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
            print (new_im)
            new_im.save("Crop.png")
            exit()

            transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground, IUV],
                                                    [False, False, True, True, True, True, True])
            data_name = ["image", "class_image", "body", "class_body", "foreground", "class_foreground", "IUV"]
            data=dict(zip(data_name, transform_output))

            data["mask"] = data["IUV"][-1,:,:]
            data["foreground"] = (data["foreground"] > 0).to(torch.long)
            data["U"] = data["IUV"][1,:,:].unsqueeze(0).to(torch.float32)/self.config["URange"]
            data["V"] = data["IUV"][0,:,:].unsqueeze(0).to(torch.float32)/self.config["VRange"]
            data.pop("IUV")


        if self.stage == 'pretrain_texture':
            data = {}
            textures = []
            texture = self.loader(os.path.join(folder, "texture", name + ".png"), mode="RGB")
            texture_tensor = F.to_tensor(texture)
            texture_size = texture_tensor.size()[1] // 4
            texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
            texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
            texture_tensor = texture_tensor
            texture_tensor = texture_tensor.contiguous().view(24 * 3, texture_size, texture_size)
            textures.append(texture_tensor)

            indexes = random.sample(list(range(0, len(self.filelists[label]))), self.config["num_texture"]-1)
            for i in indexes:
                name = self.filelists[label][i][0]
                texture = self.loader(os.path.join(folder, "texture", name+".png"), mode="RGB")
                texture_tensor = F.to_tensor(texture)
                texture_size = texture_tensor.size()[1]//4
                texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
                texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
                texture_tensor = texture_tensor.contiguous().view(24*3, texture_size, texture_size)
                textures.append(texture_tensor)

            data["texture"] = torch.stack(textures, dim=0)

        if self.stage == 'train':
            indexes = random.sample(list(range(0, len(self.filelists[label]))), 1)
            print (indexes)

            exit()
            for i in indexes:

                name = self.filelists[label][i][0]

                this_densepose_fp = os.path.join(folder, "densepose", name+".png")
                this_densepose_arr = cv2.imread(this_densepose_fp)

                this_image_fp = os.path.join(folder, 'image', name+".png")
                this_image_arr = cv2.imread(this_image_fp)
                #extract texture on the fly

                texture_ = self.GetTexture(this_image_arr, this_densepose_arr,)
                texture_tensor = F.to_tensor(texture_)

                # texture_fp = os.path.join(folder, "texture", name+".png")
                # texture_pil = self.loader(texture_fp, mode="RGB")
                # texture_pil.save('texture_not_fly.png')
                # texture_ndarray = np.asarray(texture_pil)

                # _, counts_ = np.unique(texture_, return_counts=True)
                # _, counts = np.unique(texture_ndarray,return_counts=True)

                # texture_tensor = F.to_tensor(texture_pil)
                # print (texture_tensor)
                # print (texture_)
                # print (texture_.shape, texture_.dtype)
                # print (texture_tensor.shape, texture_tensor.dtype)
                # print (texture_fp)
                # print (name)
                # print (this_densepose_fp ,this_image_fp)
                # print (torch.eq(texture_tensor, texture_))
                # print (torch.all(texture_tensor.eq(texture_)))
                # print (torch.allclose(texture_tensor, texture_))
                # exit()

                texture_size = texture_tensor.size()[1]//4
                texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
                texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
                texture_tensor = texture_tensor.contiguous().view(24*3, texture_size, texture_size)


            data["texture"] = texture_tensor.unsqueeze(0)

        data["class"] = label
        return data


class TransferDataSet(BaseDataSet):

    def __init__(self, root, src_root, config, list_name="image_list.txt"):
        super(TransferDataSet, self).__init__(config)
        self.root = root

        with open(os.path.join(root, list_name)) as f:
            filelist = f.readlines()
            filelist.sort(key=int)
            filelist = [x.strip() for x in filelist]
            self.filelist = filelist
        self.src_root = src_root

        with open(os.path.join(src_root, list_name)) as f:
            filelist = f.readlines()
            filelist.sort(key=int)
            filelist = [x.strip() for x in filelist]
            self.src_filelist = filelist

        self.size = self.config['resize']
        self.stage = self.config['phase']

    def __len__(self):
        return len(self.filelist)

    def loader(self, path, mode):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert(mode)

    def label_to_tensor(self, label):
        if isinstance(label, np.ndarray):
            return torch.from_numpy(label)
        else:
            return (F.to_tensor(label) * 255.0).type(torch.long)

    def _transform(self, images, tolabel):
        if 'resize' in self.config:
            old_size, _ = images[0].size
            size = [self.config['resize'], self.config['resize']]
            resize = transforms.Resize(size, Image.NEAREST)
            for i in range(len(images)):
                images[i] = resize(images[i])

        if 'hflip' in self.config and self.config['hflip']:
            flip = random.randint(0, 1)
        else:
            flip = 0

        if flip == 1:
            for i in range(len(images)):
                images[i] = F.hflip(images[i])

        for i in range(len(images)):
            if tolabel[i]:
                images[i] = self.label_to_tensor(images[i])
            else:
                images[i] = F.to_tensor(images[i])

        return images

    def __getitem__(self, index):

        name = self.filelist[index]
        root = self.root
        src_root = self.src_root

        image = self.loader(os.path.join(root, "image", name + ".png"), mode="RGB")
        body = self.loader(os.path.join(root, "body", name + ".png"), mode="L")
        foreground = self.loader(os.path.join(root, "segmentation", name + ".png"), mode="L")
        class_image = self.loader(os.path.join(src_root, "image", self.src_filelist[0] + ".png"), mode="RGB")
        class_foreground = self.loader(os.path.join(src_root, "segmentation", self.src_filelist[0] + ".png"), mode="L")
        class_body = self.loader(os.path.join(src_root, "body", self.src_filelist[0] + ".png"), mode="L")
        transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground], [False, False, True, True, True, True])
        data_name = ["image", "class_image", "body", "class_body", "foreground", "class_foreground"]
        data = dict(zip(data_name, transform_output))

        data["foreground"] = (data["foreground"] > 0).to(torch.long)

        textures = []

        indexes = random.sample(list(range(0, len(self.src_filelist))), self.config["num_texture"])
        for i in indexes:
            name = self.src_filelist[i]
            texture = self.loader(os.path.join(src_root, "texture", name + ".png"), mode="RGB")
            texture_tensor = F.to_tensor(texture)
            texture_size = texture_tensor.size()[1] // 4
            texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
            texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
            texture_tensor = texture_tensor
            texture_tensor = texture_tensor.contiguous().view(24 * 3, texture_size, texture_size)
            textures.append(texture_tensor)

        data["texture"] = torch.stack(textures, dim=0)

        data["class"] = 0
        return data



class RT_ReconstructDataSet(BaseDataSet):

    def __init__(self, root, config, min_sequence_len, len_ubc_dataset, list_name="image_list.txt"):
        super(RT_ReconstructDataSet, self).__init__(config)
        self.root = root

        # self.folders = glob.glob(os.path.join(root, "*"))
        self.folders = []

        for video in os.listdir(self.root):
            video_dir = os.path.join(self.root, video)
            for subject in os.listdir(video_dir):
                subject_dir = os.path.join(video_dir, subject)
                with open(os.path.join(subject_dir, list_name)) as f:
                    filelist = f.readlines()
                    if len(filelist) < min_sequence_len:
                        continue
                    else:
                        self.folders.append(subject_dir)

        self.filelist = []
        self.filelists = []
        self.len_ubc_dataset = len_ubc_dataset


        for i, folder in enumerate(self.folders):

            with open(os.path.join(folder, list_name)) as f:
                filelist = f.readlines()

                # filelist.sort(key=int)
                filelist = [(x.strip(), i+self.len_ubc_dataset) for x in filelist]


                self.filelist += filelist
                self.filelists.append(filelist)

        self.size = self.config['resize']
        self.stage = self.config['phase']

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):

        label = self.filelist[index][1]
        name = self.filelist[index][0]
        label -= self.len_ubc_dataset
        folder = self.folders[label]


        if self.stage == 'pretrain' or self.stage == 'train':

            image = self.loader(os.path.join(folder, "image", name+".jpg"), mode="RGB")
            body = self.loader(os.path.join(folder, "body", name+".png"), mode="L")
            foreground = self.loader(os.path.join(folder, "segmentation", name+".jpg"), mode="L")
            image_index = random.randrange(0, len(self.filelists[label]))
            image_name = self.filelists[label][image_index][0]
            class_image = self.loader(os.path.join(folder, "image", image_name+".jpg"), mode="RGB")
            class_foreground = self.loader(os.path.join(folder, "segmentation", image_name+".jpg"), mode="L")
            class_body = self.loader(os.path.join(folder, "body", image_name+".png"), mode="L")
            IUV = self.loader(os.path.join(folder, "densepose", name+".png") , mode="RGB")
            # IUV = imageio.imread(iuv_p)
            #
            # print (np.asarray(IUV).shape)
            # print (np.unique(np.asarray(IUV)))
            # print (np.unique(np.asarray(IUV)[:, :, 0]))
            # print (np.unique(np.asarray(IUV)[:, :, 1]))
            # print (np.unique(np.asarray(IUV)[:, :, 2]))
            # transform_iuv = self._transform([IUV], [True] )[0]
            # print (transform_iuv.shape)
            # print (np.unique(transform_iuv[0, :, :]))
            # print (np.unique(transform_iuv[1, :, :]))
            # print (np.unique(transform_iuv[2, :, :]))
            #
            # exit()
            transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground, IUV], [False, False, True, True, True, True, True])
            data_name = ["image", "class_image", "body", "class_body", "foreground", "class_foreground", "IUV"]
            data=dict(zip(data_name, transform_output))

            data["mask"] = data["IUV"][-1,:,:]
            # print (np.unique(data['mask'], return_counts=True), ' unique mask')
            # print (data["IUV"][-1, :, :].shape)
            data["foreground"] = (data["foreground"] > 0).to(torch.long)
            data["U"] = data["IUV"][1,:,:].unsqueeze(0).to(torch.float32)/self.config["URange"]
            data["V"] = data["IUV"][0,:,:].unsqueeze(0).to(torch.float32)/self.config["VRange"]
            data.pop("IUV")
            # print (np.unique(data['mask']), ' unique mask')
            # exit()

        if self.stage == 'pretrain_texture':
            data = {}
            textures = []
            texture = self.loader(os.path.join(folder, "texture", name + ".png"), mode="RGB")
            texture_tensor = F.to_tensor(texture)
            texture_size = texture_tensor.size()[1] // 4
            texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
            texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
            texture_tensor = texture_tensor
            texture_tensor = texture_tensor.contiguous().view(24 * 3, texture_size, texture_size)
            textures.append(texture_tensor)

            indexes = random.sample(list(range(0, len(self.filelists[label]))), self.config["num_texture"]-1)
            for i in indexes:
                name = self.filelists[label][i][0]
                texture = self.loader(os.path.join(folder, "texture", name+".png"), mode="RGB")
                texture_tensor = F.to_tensor(texture)
                texture_size = texture_tensor.size()[1]//4
                texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
                texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
                texture_tensor = texture_tensor.contiguous().view(24*3, texture_size, texture_size)
                textures.append(texture_tensor)

            data["texture"] = torch.stack(textures, dim=0)

        if self.stage == 'train':
            indexes = random.sample(list(range(0, len(self.filelists[label]))), 1)

            for i in indexes:
                name = self.filelists[label][i][0]
                texture = self.loader(os.path.join(folder, "texture", name+".png"), mode="RGB")
                texture_tensor = F.to_tensor(texture)
                texture_size = texture_tensor.size()[1]//4
                texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
                texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
                texture_tensor = texture_tensor.contiguous().view(24*3, texture_size, texture_size)

            data["texture"] = texture_tensor.unsqueeze(0)

        data["class"] = label
        return data


class ValidationTransferDataSet(BaseDataSet):

    def __init__(self, root, src_root, config, list_name="image_list.txt"):
        super(ValidationTransferDataSet, self).__init__(config)
        self.root = root
        self.src_root = src_root
        with open(os.path.join(root, list_name)) as f:
            filelist = f.readlines()
            filelist.sort(key=int)
            filelist = [x.strip() for x in filelist]
            self.filelist = filelist


        with open(os.path.join(src_root, list_name)) as f:
            filelist = f.readlines()

            filelist = [x.strip() for x in filelist]
            self.src_filelist = filelist

        self.size = self.config['resize']
        self.stage = self.config['phase']


    def __len__(self):
        return len(self.filelist)

    def loader(self, path, mode):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert(mode)

    def label_to_tensor(self, label):
        if isinstance(label, np.ndarray):
            return torch.from_numpy(label)
        else:
            return (F.to_tensor(label) * 255.0).type(torch.long)

    def _transform(self, images, tolabel):
        if 'resize' in self.config:
            old_size, _ = images[0].size
            size = [self.config['resize'], self.config['resize']]
            resize = transforms.Resize(size, Image.NEAREST)
            for i in range(len(images)):
                images[i] = resize(images[i])


        for i in range(len(images)):
            if tolabel[i]:
                images[i] = self.label_to_tensor(images[i])
            else:
                images[i] = F.to_tensor(images[i])

        return images

    def __getitem__(self, index):

        name = self.filelist[index]
        root = self.root
        src_root = self.src_root

        image = self.loader(os.path.join(root, "image", name + ".png"), mode="RGB")
        # print (image)
        body = self.loader(os.path.join(root, "body", name + ".png"), mode="L")
        # print (body)
        foreground = self.loader(os.path.join(root, "segmentation", name + ".png"), mode="L")
        # print (foreground)
        class_image = self.loader(os.path.join(src_root, "image", self.src_filelist[0] + ".jpg"), mode="RGB")
        # print (class_image)
        class_foreground = self.loader(os.path.join(src_root, "segmentation", self.src_filelist[0] + ".jpg"), mode="L")
        # print (class_foreground)
        class_body = self.loader(os.path.join(src_root, "body", self.src_filelist[0] + ".png"), mode="L")
        # print (class_body)
        transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground], [False, False, True, True, True, True])
        # print (transform_output)

        data_name = ["image", "class_image", "body", "class_body", "foreground", "class_foreground"]
        data = dict(zip(data_name, transform_output))

        data["foreground"] = (data["foreground"] > 0).to(torch.long)

        textures = []

        indexes = random.sample(list(range(0, len(self.src_filelist))), min(self.config["num_texture"],len(self.src_filelist)) )
        # indexes = random.sample(list(range(0, len(self.src_filelist))), self.config["num_texture"])

        for i in indexes:
            name = self.src_filelist[i]
            texture = self.loader(os.path.join(src_root, "texture", name + ".png"), mode="RGB")
            texture_tensor = F.to_tensor(texture)
            texture_size = texture_tensor.size()[1] // 4
            texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
            texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
            texture_tensor = texture_tensor
            texture_tensor = texture_tensor.contiguous().view(24 * 3, texture_size, texture_size)
            textures.append(texture_tensor)

        data["texture"] = torch.stack(textures, dim=0)

        data["class"] = 0

        return data
