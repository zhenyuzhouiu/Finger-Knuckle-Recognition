import os
import time
import torch
import torchvision.utils
from torch.autograd import Variable
from models.net_model import ResidualFeatureNet, DeConvRFNet
from data.data_factory import Factory
from torchvision import transforms
from torch.utils.data import DataLoader
from models.efficientnet import EfficientNet
from loss.loss_function import WholeImageRotationAndTranslation, ImageBlockRotationAndTranslation


def logging(msg, suc=True):
    if suc:
        print("[*] " + msg)
    else:
        print("[!] " + msg)


model_dict = {
    "RFN-128": ResidualFeatureNet(),
    "DeConvRFNet": DeConvRFNet(),
    "EfficientNet": EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
}


class Model(object):
    def __init__(self, args, writer):
        self.writer = writer
        self.batch_size = args.batch_size
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)
        self.inference, self.loss = self._build_model(args)
        self.optimizer = torch.optim.Adagrad(self.inference.parameters(), args.learning_rate)

    def _build_dataset_loader(self, args):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(args.train_path, transform=transform, valid_ext=['.bmp', '.jpg', '.JPG'], train=True,
                                losstype=args.losstype)
        logging("Successfully Load {} as training dataset...".format(args.train_path))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        examples = iter(train_loader)
        example_data, example_target = examples.next()
        example_anchor = example_data[:, 0:3, :, :]
        example_positive = example_data[:, 3:6, :, :]
        example_negative = example_data[:, 6:9, :, :]
        anchor_grid = torchvision.utils.make_grid(example_anchor)
        self.writer.add_image(tag="anchor", img_tensor=anchor_grid)
        positive_grid = torchvision.utils.make_grid(example_positive)
        self.writer.add_image(tag="positive", img_tensor=positive_grid)
        negative_grid = torchvision.utils.make_grid(example_negative)
        self.writer.add_image(tag="negative", img_tensor=negative_grid)

        return train_loader, len(train_dataset)

    def exp_lr_scheduler(self, epoch, lr_decay=0.5, lr_decay_epoch=100):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def _build_model(self, args):
        if args.model not in ["RFN-128", "DeConvRFNet", "EfficientNet"]:
            raise RuntimeError('Model not found')

        inference = model_dict[args.model].cuda()

        examples = iter(self.train_loader)
        example_data, example_target = examples.next()
        data = example_data.view(-1, 3, example_data.size(2), example_data.size(3)).cuda()
        self.writer.add_graph(inference, data[0, :, :, :].unsqueeze(0))

        if args.shifttype == "wholeimagerotationandtranslation":
            loss = WholeImageRotationAndTranslation(args.shifted_size, args.shifted_size, args.angle).cuda()
            logging("Successfully building whole image rotation and translation triplet loss")
            inference.cuda()
        elif args.shifttype == "imageblockrotationandtranslation":
            loss = ImageBlockRotationAndTranslation(args.subsize, args.dilation, args.dilation, args.angle).cuda()
            logging("Successfully building image block rotation and translation triplet loss")
            inference.cuda
        else:
            raise RuntimeError('Model loss not found')

        return inference, loss

    def triplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        for e in range(start_epoch, args.epochs + start_epoch):
            self.exp_lr_scheduler(e, lr_decay_epoch=300)
            self.inference.train()
            agg_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(self.train_loader):
                count += len(x)
                self.optimizer.zero_grad()
                x = x.cuda()

                x = Variable(x, requires_grad=False)

                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))

                # (batch_size, 12, 32, 32)
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)
                pos_fm = fms[:, 1, :, :].unsqueeze(1)
                neg_fm = fms[:, 2:, :, :].contiguous()

                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                ap_loss = self.loss(anchor_fm, pos_fm)

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)
                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer.step()

                # agg_loss += loss.data[0]
                agg_loss += loss.data
                train_loss += loss.item()

                if e % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\t {:.6f}".format(
                        time.ctime(), e, count, self.dataset_size, agg_loss / (batch_id + 1)
                    )
                    print(mesg)

                if batch_id % 5 == 0:
                    self.writer.add_scalar("loss",
                                           scalar_value=train_loss,
                                           global_step=(e * epoch_steps + batch_id))
                    train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

        self.writer.close()

    def save(self, checkpoint_dir, e):
        self.inference.eval()
        self.inference.cpu()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        torch.save(self.inference.state_dict(), ckpt_model_filename)
        self.inference.cuda()
        self.inference.train()

    def load(self, checkpoint_dir):
        self.inference.load_state_dict(torch.load(checkpoint_dir))
        self.inference.cuda()