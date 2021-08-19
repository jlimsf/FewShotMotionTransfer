from DataSet import ReconstructDataSet, RT_ReconstructDataSet, ValidationTransferDataSet
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from models.model import Model
import argparse
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import utils
import yaml
import os, cv2, traceback, shutil
import numpy as np
import random
from torchvision.utils import save_image
from models.networks import define_D

torch.manual_seed(1)
random.seed(2)
np.random.seed(3)

import wandb
# wandb.init(sync_tensorboard=True)


def validate(model, validation_dataloader, writer, device, iter, config):

    model.eval()
    iterator = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader))
    print ("validating")
    with torch.no_grad():
        running_loss = 0
        validated_labels = []
        for i, data in iterator:
            data_gpu = {key: item.to(device) for key, item in data.items()}
            validated_labels.append(np.unique(data['class'].numpy()) )

            mask, fake_image, textures, body, cordinate, losses = model(data,  "train_texture")

            for key, item in losses.items():
                losses[key] = item.mean()
                # writer.add_scalar("Validation/Loss/"+key, losses[key], iter)

            loss_G = losses.get("loss_G_L1", 0) + losses.get("loss_G_GAN", 0) + losses.get("loss_G_GAN_Feat", 0) + losses.get("loss_G_mask", 0) \
                     + losses.get("loss_texture", 0) * config['l_texture'] + losses.get("loss_coordinate", 0) * config['l_coordinate'] \
                     + losses.get("loss_mask", 0) * config['l_mask']
            running_loss += loss_G

            if i == 10:
                break
        print ("Validated on {}".format(validated_labels))
        epoch_loss = running_loss / (i+1)
        writer.add_scalar("Validation/G_Loss", epoch_loss, iter)

    model.train()


def init_embedding_matrix(model, dataloader, embedding_dir, config, device):

    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    classes = []
    print (len(dataloader))
    print ("Creating embedding directory")
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in iterator:

        data_gpu = {key: item.to(device) for key, item in data.items() if not key == 'folder'}
        folder = data['folder']

        id_data = np.unique(data["class"].detach().cpu().numpy())

        with torch.no_grad():
            print (data['texture'].shape)
            part_texture = data["texture"][:config['num_texture']]
            print (part_texture.shape)
            exit()
            b, n, c, h, w = part_texture.size()
            part_texture = part_texture.view(b * n, c, h, w)
            part_texture = torch.nn.functional.interpolate(part_texture, (self.config['texture_size'], self.config['texture_size']))
            part_texture = part_texture.view(1, b*n, c, self.config['texture_size'], self.config['texture_size'])
            self.texture_stack.data[label] = self.texture_generator(part_texture)[0].data
    exit()

    pass

def pretrain(config, writer, device_idxs=[0]):

    print (config)
    device = torch.device("cuda:" + str(device_idxs[0]))

    dataset = ReconstructDataSet(config['dataroot'], config, to_crop=True)
    dataset_RT = RT_ReconstructDataSet('/data/FSMR_data/rebecca_taylor_top_v5_256/train',
                config, min_sequence_len=2, len_ubc_dataset=len(dataset.filelists))

    joint_dataset = torch.utils.data.ConcatDataset([dataset, dataset_RT])
    joined_filelist = dataset.filelists + dataset_RT.filelists

    sampler = utils.TrainSampler(config['batchsize'], dataset.filelists)
    sampler_RT = utils.TrainSampler(config['batchsize'], dataset_RT.filelists)
    joint_sampler = utils.TrainSampler(config['batchsize'], joined_filelist)

    # data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=16, pin_memory=False, drop_last=True)
    data_loader_RT = DataLoader(dataset_RT, batch_sampler=sampler_RT, num_workers=16, pin_memory=True)


    discriminator = define_D(input_nc=3, ndf=32, netD='basic', norm='instance')
    if torch.cuda.is_available():
        discriminator.to(device)

    totol_step = 0

    model = Model(config, "train")

    # model.prepare_for_train_RT(n_class=len(joined_filelist))
    model.prepare_for_train_RT(n_class=len(dataset.filelists))
    model = model.to(device)
    model = DataParallel(model,  device_idxs)
    model.train()

    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         print (module)
    #         if hasattr(module, 'weight'):
    #             module.weight.requires_grad_(False)
    #         if hasattr(module, 'bias'):
    #             module.bias.requires_grad_(False)
    #         module.eval()
    #
    #         print (module)


    # init_embedding_matrix(model, data_loader, config['embedding_dir'], config, device)

    for epoch in trange(config['epochs']):

        iterator = tqdm(enumerate(data_loader_RT), total=len(data_loader_RT))
        for i, data in iterator:
            data_gpu = {key: item.to(device) for key, item in data.items()}



            if i % 200 <= 100:
                mask, fake_image, textures, body, cordinate, losses = model(data_gpu, "train_UV_RT")
            else:
                mask, fake_image, textures, body, cordinate, losses = model(data_gpu, "train_texture_RT")

            print (fake_image.shape)
            print (data['class_image'].shape)
            exit()

            for key, item in losses.items():
                losses[key] = item.mean()
                writer.add_scalar("Loss/"+key, losses[key], totol_step)

            if i % 200 <= 100:
                model.module.optimizer_G.zero_grad()
                # model.module.optimizer_texture_stack.zero_grad()
                # model.optimizer_G.zero_grad()
                # model.optimizer_texture_stack.zero_grad()
            else:
                model.module.optimizer_T.zero_grad()
                # model.optimizer_T.zero_grad()

            loss_G = losses.get("loss_G_L1", 0) + losses.get("loss_G_GAN", 0) + losses.get("loss_G_GAN_Feat", 0) + losses.get("loss_G_mask", 0) \
                     + losses.get("loss_texture", 0) * config['l_texture'] + losses.get("loss_coordinate", 0) * config['l_coordinate'] \
                     + losses.get("loss_mask", 0) * config['l_mask']
            loss_G.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.0, norm_type=2)

            if i % 200 <= 100:
                model.module.optimizer_G.step()
                # model.module.optimizer_texture_stack.step()
                # model.optimizer_G.step()
                # model.optimizer_texture_stack.step()
            else:
                model.module.optimizer_T.step()
                # model.optimizer_T.step()

            writer.add_scalar("Loss/G", loss_G, totol_step)

            if totol_step % config['display_freq'] == 0:
                print (loss_G.item())
                body_sum = body.sum(dim=1, keepdim=True)
                B, _, H, W = cordinate.size()
                cordinate_zero = torch.zeros((B, 1, H, W), dtype=torch.float32, device=cordinate.device)
                mask_label = torch.argmax(mask, dim=1, keepdim=True)

                cordinate_u = torch.gather(dim=1, index=mask_label, input=torch.cat((torch.zeros_like(cordinate_zero), cordinate[:, :24]), dim=1))
                cordinate_v = torch.gather(dim=1, index=mask_label, input=torch.cat((torch.zeros_like(cordinate_zero), cordinate[:, 24:]), dim=1))
                writer.add_images("Cordinate/U", utils.colorize(cordinate_u)*data_gpu["foreground"].expand_as(data["image"]).to(torch.float32), totol_step, dataformats="NCHW")
                writer.add_images("Cordinate/V", utils.colorize(cordinate_v)*data_gpu["foreground"].expand_as(data["image"]).to(torch.float32), totol_step, dataformats="NCHW")
                b, _, h, w = textures.size()
                writer.add_images("Texture", torch.clamp(textures[0].view(24, 3, h, w), 0, 1), totol_step, dataformats="NCHW")
                b, c, h, w = data_gpu["texture"][0].size()
                writer.add_images("Texture_Input", data_gpu["texture"][0].view(b, 24, 3, h, w).view(b * 24, 3, h, w), totol_step, dataformats="NCHW")
                writer.add_images("Mask/Generate", (1 - mask[:,0]).unsqueeze(1), totol_step, dataformats='NCHW')
                writer.add_images("Mask/Individual", utils.d_colorize(mask_label), totol_step, dataformats="NCHW")
                writer.add_images("Mask/Target", data["foreground"], totol_step, dataformats="NCHW")
                writer.add_images("Image/Fake", torch.clamp(fake_image, 0, 1), totol_step, dataformats="NCHW")
                writer.add_images("Image/True", data["image"] * data["foreground"].expand_as(data["image"]).to(torch.float32), totol_step, dataformats="NCHW")
                writer.add_images("Input/body", body_sum, totol_step, dataformats="NCHW")

                # validate(model, validation_dataloader, writer, device, totol_step, config)


            totol_step+=1


        #validate
        model.module.save('latest_UBC_crop')
        model.module.save(str(epoch+1)+"_UBC_crop")

        model.module.scheduler_G.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", type=int, nargs='+')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_path'], config["name"], "train"), comment=config['name'])
    pretrain(config, writer, args.device)
