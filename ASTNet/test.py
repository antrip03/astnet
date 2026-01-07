import pprint
import argparse
import tqdm
import os
import csv

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='ASTNet Kaggle Inference')

    parser.add_argument(
        '--cfg',
        help='experiment configuration filename',
        required=True,
        type=str
    )
    parser.add_argument(
        '--model-file',
        help='model parameters',
        required=True,
        type=str
    )

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    return parser.parse_args()


def main():
    args = parse_args()
    update_config(config, args)

    logger, _, _ = log_util.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = [config.GPUS[0]]

    if config.DATASET.DATASET == "ped2":
        model = get_net1(config, pretrained=False)
    else:
        model = get_net2(config, pretrained=False)

    logger.info(f'Model: {model.get_name()}')
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])

    # Load pretrained model
    state_dict = torch.load(args.model_file, map_location='cuda')
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.module.load_state_dict(state_dict)

    # Load test dataset ONLY
    test_dataset = datasets.get_test_data(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    psnr_list = inference(config, test_loader, model)

    export_kaggle_csv(psnr_list)


def inference(config, data_loader, model):
    loss_func = nn.MSELoss(reduction='none')
    model.eval()

    psnr_list = []

    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print(f'[Video {i + 1}/{len(data_loader)}]')
            psnr_video = []

            video, _ = train_util.decode_input(input=data, train=False)
            video = [f.cuda(config.GPUS[0]) for f in video]

            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                mse = torch.mean(
                    loss_func((output[0] + 1) / 2, (target[0] + 1) / 2)
                ).item()

                psnr = anomaly_util.psnr_park(mse)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)

    return psnr_list


def export_kaggle_csv(psnr_list):
    rows = []

    for vid, scores in enumerate(psnr_list, start=1):
        scores = torch.tensor(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        anomaly_scores = 1 - scores  # PSNR → anomaly

        for idx, s in enumerate(anomaly_scores, start=1):
            rows.append([f"{vid}_{idx}", float(s)])

    with open("submission_astnet.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
        writer.writerows(rows)

    print("✅ submission_astnet.csv generated")


if __name__ == '__main__':
    main()
