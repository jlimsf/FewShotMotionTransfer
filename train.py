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
import fairscale
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os, cv2, traceback, shutil
import numpy as np

import wandb
wandb.init(sync_tensorboard=True)

def validation(model, validation_loader, device, epoch, subject_name, image_size, writer):


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    folder = os.path.join('validation/')
    if not os.path.exists(folder):
        os.system("mkdir -p "+folder)

    subject_folder_name = os.path.join(folder, '/'.join(subject_name.split('/')[-3:]) )
    if not os.path.exists(subject_folder_name):

        os.makedirs(subject_folder_name)
    print ("Writing to folder: {}".format(subject_folder_name))
    out_fname = os.path.join(subject_folder_name, "{}_vid.mp4".format(epoch))
    cv2_writer = cv2.VideoWriter(out_fname, fourcc, 24, (image_size*3, image_size))
    print (out_fname)

    background = torch.ones((image_size, image_size))
    model = Model(config, "finetune")
    iter_loader = iter(validation_loader)
    model.prepare_for_finetune(next(iter_loader), background)
    model = model.to(device)
    model.background_start = model.background_start.to(device)


    vid_to_tensor = []

    with torch.no_grad():
        try:
            iterator = tqdm(enumerate(validation_loader), total=len(validation_loader))
            for i, data in iterator:
                data_gpu = {key: item.to(device) for key, item in data.items()}

                mask, fake_image, real_image, body, coordinate, texture = model(data_gpu, "inference")

                label = utils.d_colorize(data_gpu["body"]).cpu().numpy()
                B, _, H, W = coordinate.size()

                real_image = data['image'].cpu().numpy()
                fake_image = np.clip(fake_image.cpu().numpy(), 0, 1)

                outputs = np.concatenate((real_image, label, fake_image), axis=3)
                for output in outputs:
                    write_image = (output[::-1].transpose((1, 2, 0)) * 255).astype(np.uint8)

                    vid_to_tensor.append(torch.tensor(write_image))

                    cv2_writer.write(write_image)

        except Exception as e:
            print(traceback.format_exc())
            cv2_writer.release()

        cv2_writer.release()

    vid_to_tensor = torch.stack(vid_to_tensor, dim =0).unsqueeze(0)
    vid_to_tensor = vid_to_tensor.permute(0,1,4,2,3)
    
    writer.add_video(tag="Validation/Video", vid_tensor = vid_to_tensor, fps=60)


def pretrain(config, writer, device_idxs=[0]):

    print (config)
    device = torch.device("cuda:" + str(device_idxs[0]))

    # dist.init_process_group(backend='nccl', init_method="tcp://localhost:29501", rank=rank, world_size=4)

    dataset = ReconstructDataSet(config['dataroot'], config)
    dataset_RT = RT_ReconstructDataSet('/data/FSMR_data/rebecca_taylor_top/train', config)

    sampler = utils.TrainSampler(config['batchsize'], dataset.filelists)
    sampler_RT = utils.TrainSampler(config['batchsize'], dataset_RT.filelists)

    data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=16, pin_memory=True)
    data_loader_RT = DataLoader(dataset_RT, batch_sampler=sampler_RT, num_workers=0, pin_memory=True)

    #/data/FSMR_data/rebecca_taylor_top/test/000019B126/subject_1/'
    validation_dataset = ValidationTransferDataSet(root='/data/FSMR_data/top_data/train/91-2Jb8DkfS/',
                                        src_root='/data/FSMR_data/rebecca_taylor_top/test/000019B126/subject_1/',
                                        config=config)

    validation_loader = DataLoader(validation_dataset,
                                1, num_workers=4,
                                pin_memory=True,
                                shuffle=False)
    totol_step = 0
    for epoch in trange(config['epochs']):

        model = Model(config, "train")
        model.prepare_for_train(n_class=len(dataset.filelists))
        model = model.to(device)
        model = DataParallel(model,  device_idxs)
        model.train()

        iterator = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in iterator:

            data_gpu = {key: item.to(device) for key, item in data.items()}

            if i % 200 <= 100:
                mask, fake_image, textures, body, cordinate, losses = model(data_gpu, "train_UV")
            else:
                mask, fake_image, textures, body, cordinate, losses = model(data_gpu, "train_texture")

            for key, item in losses.items():
                losses[key] = item.mean()
                writer.add_scalar("Loss/"+key, losses[key], totol_step)

            if i % 200 <= 100:
                model.module.optimizer_G.zero_grad()
                model.module.optimizer_texture_stack.zero_grad()
                # optimizer_G.zero_grad()
                # optimizer_TS.zero_grad()
            else:
                model.module.optimizer_T.zero_grad()
                # optimizer_T.zero_grad()

            loss_G = losses.get("loss_G_L1", 0) + losses.get("loss_G_GAN", 0) + losses.get("loss_G_GAN_Feat", 0) + losses.get("loss_G_mask", 0) \
                     + losses.get("loss_texture", 0) * config['l_texture'] + losses.get("loss_coordinate", 0) * config['l_coordinate'] \
                     + losses.get("loss_mask", 0) * config['l_mask']

            loss_G.backward()

            if i % 200 <= 100:
                model.module.optimizer_G.step()
                model.module.optimizer_texture_stack.step()
                # optimizer_G.step()
                # optimizer_TS.step()

            else:
                model.module.optimizer_T.step()
                # optimizer_T.step()


            writer.add_scalar("Loss/G", loss_G, totol_step)

            if totol_step % config['display_freq'] == 0:
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

            totol_step+=1

            break


        #validate



        model.module.save('latest_train')
        model.module.save(str(epoch+1)+"_train")

        model.module.scheduler_G.step()
        print ("Validation")
        validation(model, validation_loader, device,epoch, subject_name = validation_loader.dataset.src_root, image_size=config['resize'], writer=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", type=int, nargs='+')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_path'], config["name"], "train"), comment=config['name'])
    pretrain(config, writer, args.device)
