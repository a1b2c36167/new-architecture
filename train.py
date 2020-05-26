import argparse
import time

import torch.optim as optim
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
#from models_all import *
from models import *
from utils.datasets import *
from utils.utils import *

# Hyperparameters
hyp = {'k': 8.4875,  # loss multiple
       'xy': 0.079756,  # xy loss fraction
       'wh': 0.010461,  # wh loss fraction
       'cls': 0.02105,  # cls loss fraction
       'conf': 0.88873,  # conf loss fraction
       'iou_t': 0.10,  # iou target-anchor training threshold
       'lr0': 0.0001,  # initial learning rate #before 0.001 / after 0.0001
       'lrf': -5,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       }


def train(
        cfg,
        cfg2,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        data_augment=False,
        freeze_backbone=False,
        transfer=False  # Transfer learning (train only YOLO layers)
):
    init_seeds()
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
        opt.num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = DarknetPlus(cfg, cfg2, img_size).to(device)

    # loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = int(model.model1.module_defs[model.model1.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(weights + 'yolov3.pt', map_location=device)
            model.model1.load_state_dict({k: v for k, v in chkpt['model1'].items() if v.numel() > 1 and v.shape[0] != 255}, strict=False)
            for p in model.model1.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.model1.load_state_dict(chkpt['model1'])
            model.model2.load_state_dict(chkpt['model2'])
            model.model3.load_state_dict(chkpt['model3'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model.model1, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model.model1, weights + 'darknet53.conv.74')
            #cutoff = load_darknet_weights(model.model1, weights + 'yolov3.weights.74')

    # Scheduler (reduce lr at epochs 218, 245, i.e. batches 400k, 450k)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[218, 245],
                                               gamma=0.1,
                                               last_epoch=start_epoch - 1)

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size=img_size, augment=data_augment)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn,
                            sampler=None)

    # Start training
    t = time.time()
    model.model1.hyp = hyp  # attach hyperparameters to model
    model.model2.hyp = hyp  # attach hyperparameters to model
    #model_info(model.model1)
    #model_info(model.model2)
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    os.remove('train_batch0.jpg') if os.path.exists('train_batch0.jpg') else None
    os.remove('test_batch0.jpg') if os.path.exists('test_batch0.jpg') else None
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 15) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'classify', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.model1.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss1 = torch.zeros(5).to(device)  # mean losses
        mloss2 = torch.zeros(5).to(device)  # mean losses
        mloss3 = torch.zeros(1).to(device)
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            targets, LS_targets, LS_classify_targets = model.transform_GT_with_targets(targets)
            pred1, pred2, pred3 = model(imgs, targets, LS_targets)
            
            # Compute loss
            loss1, loss_items1 = compute_loss(pred1, targets, model.model1)
            loss2, loss_items2 = compute_loss(pred2, LS_targets, model.model2)
            if pred3 is not None:
                #print("================",pred3.size(),LS_classify_targets[:, 1].to(device=device, dtype=torch.int64).size())
                loss3 = criterion(pred3, LS_classify_targets[:, 1].to(device=device, dtype=torch.int64))
                loss = loss1 + loss2 + loss3
            else:
                loss = loss1 + loss2
            #print(float(loss))
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Update running mean of tracked metrics
            mloss1 = (mloss1 * i + loss_items1) / (i + 1)
            mloss2 = (mloss2 * i + loss_items2) / (i + 1)
            mloss3 = (mloss3 * i + loss3) / (i + 1)

            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 15) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss1, len(targets), *mloss2, len(LS_targets), mloss3, len(LS_classify_targets), time.time() - t)
            t = time.time()
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 40 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 5)) or epoch == epochs - 1:
            with torch.no_grad():
                results = test.test(cfg, cfg2, data_cfg, batch_size=batch_size, img_size=img_size, model=model, conf_thres=0.1)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 13 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[-1]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        save = True and not opt.nosave
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model1': model.model1.state_dict(),
                     'mAP1': results[2],
                     'model2': model.model2.state_dict(),
                     'mAP2': results[6],
                     'model3': model.model3.state_dict(),
                     'mAP3': results[10],
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/new_yolov3.cfg', help='cfg file path')
    parser.add_argument('--cfg2', type=str, default='cfg/new2_yolov3-tiny.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/tlr.data', help='.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--data-augment', action='store_true', help='random augmentation to the input images')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=0, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    # Train
    results = train(
        opt.cfg,
        opt.cfg2,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume or opt.transfer,
        transfer=opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
        data_augment=opt.data_augment
    )
