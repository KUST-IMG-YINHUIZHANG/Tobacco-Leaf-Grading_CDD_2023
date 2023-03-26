import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model

from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    train_images_path, train_images_label, train_images_sheet, val_images_path, val_images_label, val_images_sheet = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              images_sheet=train_images_sheet,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            images_sheet=val_images_sheet,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # load pre-train
    model = create_model(num_classes=args.num_classes).to(device)
    model_dict = model.state_dict()
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)

        weights_dict = torch.load(args.weights, map_location=device)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        model_dict.update(weights_dict)

        weights_dict1 = torch.load(args.weights1, map_location=device)["model"]
        for k in list(weights_dict1.keys()):
            if "head" in k:
                del weights_dict1[k]
        model_dict.update(weights_dict1)
        print(model.load_state_dict(model_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    max_train_acc = 0
    max_val_acc = 0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        if train_acc>=max_train_acc:
            max_train_acc = train_acc
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        if val_acc>=max_val_acc:
            max_val_acc = train_acc

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    print("Max train acc:", max_train_acc)
    print("Max val acc:", max_val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                        default="./tobacco_photos_0705")
    parser.add_argument('--weights', type=str, default='./mobilenet_v2-b0353104.pth',
                        help='initial weights path')
    parser.add_argument('--weights1', type=str, default='./swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
