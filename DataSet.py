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
from io import BytesIO
import datetime

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

    def __init__(self, root, config, to_crop = True, list_name="image_list.txt"):
        super(ReconstructDataSet, self).__init__(config)
        self.root = root

        self.folders = glob.glob(os.path.join(root, "*"))
        self.folders.sort()

        self.filelist = []
        self.filelists = []
        self.to_crop = to_crop

        # self.folders = self.folders[:1]

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

    def _transform(self, images, tolabel, to_normalize, crop_params, to_flip, return_tensor=True):

        i,j,h,w = crop_params
        normalize_function = transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
        #random resize crop

        if 'resize' in self.config:
            old_size, _ = images[0].size
            size = [self.config['resize'], self.config['resize']]
            resize = transforms.Resize(size, Image.NEAREST)

            if self.to_crop == True:
                coin = random.randint(1, 2)
            else:
                coin = 2 #do not crop

            for i in range(len(images)):
                this_im = images[i]
                if coin == 1:
                    this_im = F.crop(this_im, i, j, h, w)

                resized = resize(this_im)
                # resized = new_im
                # resized.save('{}.png'.format(i))
                images[i] = resized

        if to_flip==1:
            for i in range(len(images)):
                images[i] = F.hflip(images[i])

        if return_tensor:
            for i in range(len(images)):
                if tolabel[i]:
                    images[i] = self.label_to_tensor(images[i])
                else:
                    images[i] = F.to_tensor(images[i])
                    if to_normalize:
                        images[i] = normalize_function(images[i])
            return images
        else:
            return images

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def GetTexture(self, im, IUV):
        '''
        This indexing below is different than the original author's code
        '''

        U = IUV[:, :, 1]
        V = IUV[:, :, 0]

        Texture = np.zeros((24, 128, 128, 3), dtype=np.uint8)
        for PartInd in range(1, 25):
            tex = Texture[PartInd - 1, :, :, :].squeeze()
            '''
            This indexing below is different than the original author's code
            '''
            x, y = np.where(IUV[:, :, 2] == PartInd)
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
            image = self.loader(os.path.join(folder, "image", name+".jpg"), mode="RGB")
            body = self.loader(os.path.join(folder, "body", name+".png"), mode="L")
            foreground = self.loader(os.path.join(folder, "segmentation", name+".png"), mode="L")

            #Targets
            image_index = random.randrange(0, len(self.filelists[label]))
            image_name = self.filelists[label][image_index][0]
            class_image = self.loader(os.path.join(folder, "image", image_name+".jpg"), mode="RGB")
            class_foreground = self.loader(os.path.join(folder, "segmentation", image_name+".png"), mode="L")
            class_body = self.loader(os.path.join(folder, "body", image_name+".png"), mode="L")
            IUV = self.loader(os.path.join(folder, "densepose", name+".png"), mode="RGB")

            i,j,h,w = self.get_params(image, output_size=(self.size//1.5, self.size//1.5) )

            if 'hflip' in self.config and self.config['hflip']:
                flip_val = random.randint(0, 1)
            else:
                flip_val = 0


            transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground, IUV],
                                                    tolabel=[False, False, True, True, True, True, True],
                                                    to_normalize = [True, True, False, False, False, False, False],
                                                    crop_params = [i,j,h,w], to_flip = flip_val)
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

            this_densepose_fp = os.path.join(folder, "densepose", name+".png")
            this_densepose_pil = self.loader(this_densepose_fp, mode='RGB')

            this_image_fp = os.path.join(folder, 'image', name+".png")
            this_image_pil = self.loader(this_image_fp, mode="RGB")
            #extract texture on the fly

            if 'hflip' in self.config and self.config['hflip']:
                flip_val = random.randint(0, 1)
            else:
                flip_val = 0

            i,j,h,w = self.get_params(this_image_pil, output_size=(self.size//1.5, self.size//1.5) )
            [transforms_densepose, transforms_image] = \
                self._transform([this_densepose_pil, this_image_pil],
                                [True, False], [False, True],
                                crop_params = [i,j,h,w], to_flip=flip_val, return_tensor=False)

            texture_ = self.GetTexture(np.asarray(transforms_image), np.asarray(transforms_densepose))
            texture_tensor = F.to_tensor(texture_)


            texture_size = texture_tensor.size()[1] // 4
            texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
            texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
            texture_tensor = texture_tensor
            texture_tensor = texture_tensor.contiguous().view(24 * 3, texture_size, texture_size)
            textures.append(texture_tensor)

            data["image"] = self._transform([ this_image_pil], [ False], [True],
                            crop_params = [i,j,h,w], to_flip=flip_val, return_tensor=True)[0]

            indexes = random.sample(list(range(0, len(self.filelists[label]))), self.config["num_texture"]-1)
            for i in indexes:
                name = self.filelists[label][i][0]
                # texture = self.loader(os.path.join(folder, "texture", name+".png"), mode="RGB")
                this_densepose_fp = os.path.join(folder, "densepose", name+".png")
                this_densepose_pil = self.loader(this_densepose_fp, mode='RGB')

                this_image_fp = os.path.join(folder, 'image', name+".png")
                this_image_pil = self.loader(this_image_fp, mode="RGB")
                #extract texture on the fly

                [transforms_densepose, transforms_image] = \
                    self._transform([this_densepose_pil, this_image_pil],
                    [True, False], [False, True], crop_params = [i,j,h,w], to_flip=flip_val, return_tensor=False )

                texture_ = self.GetTexture(np.asarray(transforms_image), np.asarray(transforms_densepose))
                texture_tensor = F.to_tensor(texture_)
                # texture_tensor = F.to_tensor(texture)
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

                this_densepose_fp = os.path.join(folder, "densepose", name+".png")
                this_densepose_pil = self.loader(this_densepose_fp, mode='RGB')

                this_image_fp = os.path.join(folder, 'image', name+".jpg")
                this_image_pil = self.loader(this_image_fp, mode="RGB")
                #extract texture on the fly

                [transforms_densepose, transforms_image] = \
                    self._transform([this_densepose_pil, this_image_pil],
                    [True, False], [False, False], crop_params = [i,j,h,w], to_flip=flip_val, return_tensor=False )


                texture_ = self.GetTexture(np.asarray(transforms_image), np.asarray(transforms_densepose))
                texture_tensor = F.to_tensor(texture_)

                texture_size = texture_tensor.size()[1]//4
                texture_tensor = texture_tensor.view(-1, 4, texture_size, 6, texture_size)
                texture_tensor = texture_tensor.permute(1, 3, 0, 2, 4)
                texture_tensor = texture_tensor.contiguous().view(24*3, texture_size, texture_size)


            data["texture"] = texture_tensor.unsqueeze(0)

        data["class"] = label

        return data

class OriginalReconstructDataSet(BaseDataSet):

    def __init__(self, root, config, list_name="image_list.txt"):
        super(OriginalReconstructDataSet, self).__init__(config)
        self.root = root

        self.folders = glob.glob(os.path.join(root, "*"))
        self.folders.sort()

        self.filelist = []
        self.filelists = []


        for i, folder in enumerate(self.folders):

            with open(os.path.join(folder, list_name)) as f:
                filelist = f.readlines()
                # filelist.sort(key=int)
                filelist = [(x.strip(), i) for x in filelist]
                print (filelist)
                self.filelist += filelist
                self.filelists.append(filelist)

        self.size = self.config['resize']
        self.stage = self.config['phase']

    def _transform(self, images, tolabel,  return_tensor=True):


        if 'resize' in self.config:
            old_size, _ = images[0].size
            size = [self.config['resize'], self.config['resize']]
            resize = transforms.Resize(size, Image.NEAREST)

            for i in range(len(images)):
                this_im = images[i]
                resized = resize(this_im)
                images[i] = resized

        if return_tensor:
            for i in range(len(images)):
                if tolabel[i]:
                    images[i] = self.label_to_tensor(images[i])
                else:
                    images[i] = F.to_tensor(images[i])

            return images
        else:
            return images


    def GetTexture(self, im, IUV):
        '''
        This indexing below is different than the original author's code
        '''

        U = IUV[:, :, 1]
        V = IUV[:, :, 0]

        Texture = np.zeros((24, 128, 128, 3), dtype=np.uint8)
        for PartInd in range(1, 25):
            tex = Texture[PartInd - 1, :, :, :].squeeze()
            '''
            This indexing below is different than the original author's code
            '''
            x, y = np.where(IUV[:, :, 2] == PartInd)
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


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):

        label = self.filelist[index][1]
        name = self.filelist[index][0]
        folder = self.folders[label]

        if self.stage == 'pretrain' or self.stage == 'train':
            image = self.loader(os.path.join(folder, "image", name+".png"), mode="RGB")
            body = self.loader(os.path.join(folder, "body", name+".png"), mode="L")
            foreground = self.loader(os.path.join(folder, "segmentation", name+".png"), mode="L")

            image_index = random.randrange(0, len(self.filelists[label]))
            image_name = self.filelists[label][image_index][0]
            class_image = self.loader(os.path.join(folder, "image", image_name+".png"), mode="RGB")
            class_foreground = self.loader(os.path.join(folder, "segmentation", image_name+".png"), mode="L")

            class_body = self.loader(os.path.join(folder, "body", image_name+".png"), mode="L")
            densepose_fp =  os.path.join(folder, "densepose", name+".png")
            IUV = self.loader(densepose_fp, mode="RGB")

            # print (np.unique(np.asarray(IUV)[:,:,0]))
            # print (np.unique(np.asarray(IUV)[:,:,1]))
            # print (np.unique(np.asarray(IUV)[:,:,2]))
            transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground, IUV],
                                                    [False, False, True, True, True, True, True])


            data_name = ["image", "class_image", "body", "class_body", "foreground", "class_foreground", "IUV"]
            data=dict(zip(data_name, transform_output))

            # PILtoIM = transforms.ToPILImage()
            time_str = str(datetime.datetime.now())
            # # print(densepose_fp)
            # PILtoIM(data['class_foreground']).save('{}_c_foreground.png'.format(time_str))
            # PILtoIM(data['foreground']).save('{}_foregroundpng'.format(time_str))
            # exit()

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
            for i in indexes:

                name = self.filelists[label][i][0]

                this_densepose_fp = os.path.join(folder, "densepose", name+".png")
                this_densepose_pil = self.loader(this_densepose_fp, mode='RGB')

                this_image_fp = os.path.join(folder, 'image', name+".png")
                this_image_pil = self.loader(this_image_fp, mode="RGB")
                #extract texture on the fly

                [transforms_densepose, transforms_image] = \
                    self._transform([this_densepose_pil, this_image_pil],
                    [True, False], return_tensor=False )


                texture_ = self.GetTexture(np.asarray(transforms_image), np.asarray(transforms_densepose))

                # cv2.imwrite('texture_{}.png'.format(time_str), texture_ )
                # exit()
                texture_tensor = F.to_tensor(texture_)
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
            # filelist.sort(key=int)
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

    def _transform(self, images, tolabel,  return_tensor=True):


        if 'resize' in self.config:
            old_size, _ = images[0].size
            size = [self.size, self.size]
            resize = transforms.Resize(size, Image.NEAREST)

            for i in range(len(images)):
                this_im = images[i]
                resized = resize(this_im)
                images[i] = resized

        if return_tensor:
            for i in range(len(images)):
                if tolabel[i]:
                    images[i] = self.label_to_tensor(images[i])
                else:
                    images[i] = F.to_tensor(images[i])

            return images
        else:
            return images


    def GetTexture(self, im, IUV):
        '''
        This indexing below is different than the original author's code
        '''

        U = IUV[:, :, 1]
        V = IUV[:, :, 0]

        Texture = np.zeros((24, 128, 128, 3), dtype=np.uint8)
        for PartInd in range(1, 25):
            tex = Texture[PartInd - 1, :, :, :].squeeze()
            '''
            This indexing below is different than the original author's code
            '''
            x, y = np.where(IUV[:, :, 2] == PartInd)
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

        name = self.filelist[index]
        root = self.root
        src_root = self.src_root

        image = self.loader(os.path.join(root, "image", name + ".png"), mode="RGB")
        body = self.loader(os.path.join(root, "body", name + ".png"), mode="L")
        foreground = self.loader(os.path.join(root, "segmentation", name + ".png"), mode="L")
        class_image = self.loader(os.path.join(src_root, "image", self.src_filelist[0] + ".png"), mode="RGB")

        class_foreground = self.loader(os.path.join(src_root, "segmentation", self.src_filelist[0] + ".png"), mode="L")
        class_body = self.loader(os.path.join(src_root, "body", self.src_filelist[0] + ".png"), mode="L")
        IUV = self.loader(os.path.join(src_root, "densepose", self.src_filelist[0]+".png"), mode='RGB')


        transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground, IUV],
                                            [False, False, True, True, True, True, True])

        data_name = ["image", "class_image", "body", "class_body", "foreground", "class_foreground", "IUV"]
        data = dict(zip(data_name, transform_output))

        # PILtoIM = transforms.ToPILImage(mode='RGB')
        time_str = str(datetime.datetime.now())
        # print(densepose_fp)
        # PILtoIM(data['IUV']).save('{}_iuv.png'.format(time_str))
        # PILtoIM(data['class_image']).save('{}_class_image.png'.format(time_str))
        # exit()

        data["foreground"] = (data["foreground"] > 0).to(torch.long)

        textures = []

        indexes = random.sample(list(range(0, len(self.src_filelist))), self.config["num_texture"])
        for i in indexes:
            name = self.src_filelist[i]


            this_densepose_fp = os.path.join(src_root, "densepose", name+".png")
            this_densepose_pil = self.loader(this_densepose_fp, mode='RGB')

            this_image_fp = os.path.join(src_root, 'image', name+".png")
            this_image_pil = self.loader(this_image_fp, mode="RGB")
            #extract texture on the fly

            [transforms_densepose, transforms_image] = \
                self._transform([this_densepose_pil, this_image_pil],
                [True, False], return_tensor=False )


            texture_ = self.GetTexture(np.asarray(transforms_image), np.asarray(transforms_densepose))
            cv2.imwrite('texture_{}.png'.format(time_str), texture_ )
            exit()
            texture_tensor = F.to_tensor(texture_)

            # texture = self.loader(os.path.join(src_root, "texture", name + ".png"), mode="RGB")
            # texture_tensor = F.to_tensor(texture)
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
        body = self.loader(os.path.join(root, "body", name + ".png"), mode="L")
        foreground = self.loader(os.path.join(root, "segmentation", name + ".png"), mode="L")
        class_image = self.loader(os.path.join(src_root, "image", self.src_filelist[0] + ".jpg"), mode="RGB")
        class_foreground = self.loader(os.path.join(src_root, "segmentation", self.src_filelist[0] + ".jpg"), mode="L")
        class_body = self.loader(os.path.join(src_root, "body", self.src_filelist[0] + ".png"), mode="L")
        transform_output = self._transform([image, class_image, body, class_body, foreground, class_foreground], [False, False, True, True, True, True])

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
