from DataSet import ReconstructDataSet
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from models.model import Model
import argparse
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import os, random
import numpy as np
import utils
import yaml

import wandb
# wandb.init(sync_tensorboard=True)

torch.manual_seed(1)
random.seed(2)
np.random.seed(3)

def pretrain(config, writer, device_idxs=[0]):

    data_loader = DataLoader(ReconstructDataSet(config['dataroot'], config),
                    batch_size=config['batchsize'], num_workers=0,
                    pin_memory=False, shuffle=True)
    print (data_loader)



    model = Model(config, "pretrain")
    model.prepare_for_pretrain()
    device = torch.device("cuda:" + str(device_idxs[0]))
    model = model.to(device)
    model = DataParallel(model, device_idxs)
    model.train()

    # wandb.watch(model.module.generator, log_freq=1, log='all')


    totol_step = 0
    for epoch in trange(config['epochs']):
        iterator = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in iterator:

            # if i < 600: continue

            for k,v in data.items():
                # v = torch.tensor([1, 2, np.nan])
                if k == 'class': continue
                if torch.isnan(v).any():
                    print ("Found nan in {} , {}".format(i, k))

            data_gpu = {key: item.to(device) for key, item in data.items()}

            loss_mask, loss_coordinate, mask, coordinate, body, label_mask = model(data_gpu, "pretrain")


            loss_mask = loss_mask.mean()
            loss_coordinate = loss_coordinate.mean()
            model.module.optimizer_G.zero_grad()
            (loss_mask+loss_coordinate).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.0, norm_type=2)

            model.module.optimizer_G.step()
            writer.add_scalar("Loss/Mask", loss_mask, totol_step)
            writer.add_scalar("Loss/coordinate", loss_coordinate, totol_step)

            print (i ,loss_coordinate, loss_mask)

            if totol_step % config['display_freq'] == 0:

                for i in range(mask.size()[1]):
                #
                #     # wandb.Image(_)
                #     wandb.log({"Mask/"+str(i).zfill(2): wandb.Image(_) for _ in mask[:,i] })
                #     if i<24:
                #         wandb.log({"U/"+str(i+1).zfill(2): wandb.Image(_) for _ in utils.colorize(coordinate[:,i].unsqueeze(1)) }),
                #         wandb.log({"V/"+str(i+1).zfill(2): wandb.Image(_) for _ in utils.colorize(coordinate[:,i+24].unsqueeze(1)) }),

                    writer.add_images("Mask/"+str(i).zfill(2), mask[:,i].unsqueeze(1), totol_step, dataformats="NCHW")
                    # if i<24:
                    #     writer.add_images("U/"+str(i+1).zfill(2), coordinate[:,i].unsqueeze(1), totol_step, dataformats="NCHW")
                    #     writer.add_images("V/"+str(i+1).zfill(2), coordinate[:,i+24].unsqueeze(1), totol_step, dataformats="NCHW")
                body_sum = body.sum(dim=1, keepdim=True)
                body_sum = (body_sum-body_sum.min())/(body_sum.max()-body_sum.min())
                writer.add_images("Target/Mask", utils.d_colorize(data["mask"].unsqueeze(1)), totol_step, dataformats="NCHW")
                writer.add_images("Target/U", utils.colorize(data["U"]), totol_step, dataformats="NCHW")
                writer.add_images("Target/V", utils.colorize(data["V"]), totol_step, dataformats="NCHW")
                writer.add_images("Input/body", body_sum, totol_step, dataformats="NCHW")
                writer.add_images("Target/Image", data["image"], totol_step, dataformats="NCHW")
                writer.add_images("coordinate/U", utils.colorize(torch.sum(coordinate[:,:24]*label_mask[:,:24], dim=1, keepdim=True)), totol_step, dataformats="NCHW")
                writer.add_images("coordinate/V", utils.colorize(torch.sum(coordinate[:,24:]*label_mask[:,24:], dim=1, keepdim=True)), totol_step, dataformats="NCHW")

            totol_step+=1

        model.module.save('latest')
        model.module.save(str(epoch+1))



        model.module.scheduler_G.step()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", type=int, nargs='+')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_path'], config["name"], "pretrain"), comment=config['name'])
    pretrain(config, writer, args.device)
