# =========================================================
# @ Main File for PolyU Project: Online Contactless Palmprint
#   Identification using Deep Learning
# =========================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import shutil
import scipy.misc
import datetime
from models.model import Model
from torch.utils.tensorboard import SummaryWriter


def build_parser():
    parser = argparse.ArgumentParser()

    # Checkpoint Options
    parser.add_argument('--checkpoint_dir', type=str,
                        dest='checkpoint_dir', default='./checkpoint/RFNet/')
    parser.add_argument('--db_prefix', dest='db_prefix', default='fkv3(yolov5-184-208)-105-221')
    parser.add_argument('--checkpoint_interval', type=int, dest='checkpoint_interval', default=20)

    # Dataset Options
    parser.add_argument('--train_path', type=str, dest='train_path',
                        default='./dataset/PolyUKnuckleV3/yolov5/184_208/Session_1/105-221/')

    # Training Strategy
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=4)
    parser.add_argument('--epochs', type=int, dest='epochs', default=3000)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-3)

    # Training Logging Interval
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=1)
    # Pre-defined Options
    parser.add_argument('--shifttype', type=str, dest='shifttype', default='wholeimagerotationandtranslation')
    parser.add_argument('--alpha', type=float, dest='alpha', default=20)
    parser.add_argument('--model', type=str, dest='model', default="RFNet")
    parser.add_argument('--input_size', type=int, dest='input_size', default=(184, 208))
    parser.add_argument('--horizontal_size', type=int, dest='horizontal_size', default=4)
    parser.add_argument('--vertical_size', type=int, dest='vertical_size', default=8)
    parser.add_argument('--block_size', type=int, dest="block_size", default=8)
    parser.add_argument('--rotate_angle', type=int, dest="rotate_angle", default=4)

    # fine-tuning
    parser.add_argument('--start_ckpt', type=str, dest='start_ckpt', default="./checkpoint/RFNet/fkv1_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle5-a10-s4_2022-06-12-17-06/ckpt_epoch_360.pth")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    this_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        "{}_{}-{}-lr{}-subs{}-angle{}-a{}-hs{}_vs{}_{}".format(
            args.db_prefix,
            args.model,
            args.shifttype,
            float(args.learning_rate),
            int(args.block_size),
            int(args.rotate_angle),
            int(args.alpha),
            int(args.horizontal_size),
            int(args.vertical_size),
            this_datetime
        )
    )

    logdir = os.path.join(args.checkpoint_dir, 'runs')


    print("[*] Target Checkpoint Path: {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    print("[*] Target Logdir Path: {}".format(logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    hyper_parameter = os.path.join(args.checkpoint_dir, 'hyper_parameter.txt')
    with open(hyper_parameter, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    writer = SummaryWriter(log_dir=logdir)
    model_ = Model(args, writer=writer)
    model_.triplet_train(args)


if __name__ == "__main__":
    main()

