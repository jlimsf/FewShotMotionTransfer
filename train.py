from DataSet import ReconstructDataSet
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

import wandb
wandb.init(sync_tensorboard=True)

def pretrain(config, writer, device_idxs=[0]):

    world_size = 4
    print (config)

    # dist.init_process_group(backend='nccl', init_method="tcp://localhost:29501", rank=rank, world_size=4)

    dataset = ReconstructDataSet(config['dataroot'], config)
    sampler = utils.TrainSampler(config['batchsize'], dataset.filelists)

    data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=16, pin_memory=True)

    model = Model(config, "train")
    model.prepare_for_train(n_class=len(dataset.filelists))
    device = torch.device("cuda")
    model = model.to(device)
    model = DataParallel(model, [0,1,2,3])
    # model = model.to(rank)
    model.train()

    # optimizer_G = model.optimizer_G
    # optimizer_TS = model.optimizer_texture_stack
    # optimizer_T = model.optimizer_T

    # base_optimizer_arguments = {'lr':0.0002, 'betas':(0.5,0.999)}


    # optimizer_G = OSS(params=optimizer_G.param_groups, optim=torch.optim.Adam,  **base_optimizer_arguments)
    # optimizer_TS = OSS(params=optimizer_TS.param_groups, optim=torch.optim.Adam, **base_optimizer_arguments)
    # optimizer_T = OSS(params=optimizer_T.param_groups, optim=torch.optim.Adam,  **base_optimizer_arguments)

    # model = ShardedDDP(model, [optimizer_G, optimizer_TS, optimizer_T])
    # model = model.train()
    # scaler = torch.cuda.amp.GradScaler(enabled=True)

    totol_step = 0
    for epoch in trange(config['epochs']):
        iterator = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in iterator:

            data_gpu = {key: item.to(device) for key, item in data.items()}
            # data_gpu = data
            # # torch.cuda.empty_cache()
            # with torch.cuda.amp.autocast(enabled=True):

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



        model.module.save('latest_train')
        model.module.save(str(epoch+1)+"_train")

        model.module.scheduler_G.step()


    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", type=int, nargs='+')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_path'], config["name"], "train"), comment=config['name'])
    pretrain(config, writer, args.device)

    # mp.spawn(
    #     pretrain,
    #     args=(
    #         config,
    #         args.device
    #     ),
    #     nprocs=4,
    #     join=True,
    # )
