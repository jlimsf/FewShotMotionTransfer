

def inference(model, config, device_idxs=[0]):
    config['phase'] = 'inference'
    config['hflip'] = False
    dataset = TransferDataSet(config['target_root'], config['source_root'], config)
    data_loader = DataLoader(dataset, batch_size=config['batchsize'], num_workers=4, pin_memory=True, shuffle=False)

    device = torch.device("cuda:" + str(device_idxs[0]))
    image_size = config['resize']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    folder = os.path.join(config["output_folder"], config["name"])
    if not os.path.exists(folder):
        os.system("mkdir -p "+folder)

    print ("Writing to folder: {}".format(folder))
    writer = cv2.VideoWriter(os.path.join(folder, config['output_name']), fourcc, 24, (image_size*3, image_size))
    print (config['output_name'])
    with torch.no_grad():
        try:
            iterator = tqdm(enumerate(data_loader), total=len(data_loader))
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
                    writer.write(write_image)

        except Exception as e:
            print(traceback.format_exc())
            writer.release()

        writer.release()
