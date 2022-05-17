# =========================================================
# @ Main File for PolyU Project: Online Contactless Palmprint
#   Identification using Deep Learning
# =========================================================

import argparse
import os
import shutil
import scipy.misc
import datetime
from models.model import Model
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def build_parser():
    parser = argparse.ArgumentParser()

    # Checkpoint Options
    parser.add_argument('--logdir', type=str, dest='logdir', default='./runs/checkstnetandbinary/')

    parser.add_argument('--checkpoint_dir', type=str,
                        dest='checkpoint_dir', default='./checkpoint/checkstnetandbinary/')
    parser.add_argument('--db_prefix', dest='db_prefix', default='fkv3-session2')
    parser.add_argument('--checkpoint_interval', type=int, dest='checkpoint_interval', default=20)

    # Dataset Options
    parser.add_argument('--train_path', type=str, dest='train_path',
                        default='./dataset/PolyUKnuckleV3/Session_2_128/')

    # Training Strategy
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=4)
    parser.add_argument('--epochs', type=int, dest='epochs', default=6000)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-4)

    # Training Logging Interval
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=1)
    # Pre-defined Options
    parser.add_argument('--shifttype', type=str, dest='shifttype', default='hammingdistance')
    parser.add_argument('--alpha', type=float, dest='alpha', default=17)
    parser.add_argument('--model', type=str, dest='model', default="ConvNetEfficientSTNetBinaryConvNet")
    parser.add_argument('--input_size', type=int, dest='input_size', default=128)
    parser.add_argument('--shifted_size', type=int, dest='shift_size', default=4)
    parser.add_argument('--block_size', type=int, dest="block_size", default=8)
    parser.add_argument('--rotate_angle', type=int, dest="rotate_angle", default=5)

    # fine-tuning
    parser.add_argument('--start_ckpt', type=str, dest='start_ckpt', default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    this_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        "{}_{}-{}-lr{}-subs{}-angle{}-a{}-s{}_{}".format(
            args.db_prefix,
            args.model,
            args.shifttype,
            float(args.learning_rate),
            int(args.block_size),
            int(args.rotate_angle),
            int(args.alpha),
            int(args.shift_size),
            this_datetime
        )
    )

    args.logdir = os.path.join(
        args.logdir,
        "{}_{}-{}-lr{}-subs{}-angle{}-a{}-s{}_{}".format(
            args.db_prefix,
            args.model,
            args.shifttype,
            float(args.learning_rate),
            int(args.block_size),
            int(args.rotate_angle),
            int(args.alpha),
            int(args.shift_size),
            this_datetime
        )
    )

    print("[*] Target Checkpoint Path: {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    print("[*] Target Logdir Path: {}".format(args.logdir))
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    hyper_parameter = os.path.join(args.checkpoint_dir, 'hyper_parameter.txt')
    with open(hyper_parameter, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    writer = SummaryWriter(log_dir=args.logdir)
    model_ = Model(args, writer=writer)
    model_.triplet_train(args)


if __name__ == "__main__":
    main()

