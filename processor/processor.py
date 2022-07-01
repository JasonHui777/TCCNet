import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.util import Timer


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, device, resume):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    start_epoch = 1
    if resume:
        start_epoch = resume.split("/")[-1].split("_")[-1].split(".")[0]
        start_epoch = int(start_epoch) + 1
        param_dict = torch.load(resume, map_location='cuda')
        if "model" in param_dict.keys():
            model.load_model_param(param_dict["model"])
        if "optimizer" in param_dict.keys():
            optimizer.load_state_dict(param_dict["optimizer"])
        if "optimizer_center" in param_dict.keys():
            optimizer_center.load_state_dict(param_dict["optimizer_center"])
        if "scheduler" in param_dict.keys():
            scheduler.load_state_dict(param_dict["scheduler"])

    loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    triplet_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # CNN 进行冻结
    if cfg.SOLVER.FREEZE_PARAM:
        for p in model.base.trans_1.parameters():
            p.requires_grad = False
        for i in range(2, 13):
            model_base = model.base
            conv_trans = getattr(model_base, "conv_trans_" + str(i))
            conv_trans = conv_trans.trans_block
            for p in conv_trans.parameters():
                p.requires_grad = False

    timer = Timer()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    flag = False
    # train
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        id_loss_meter.reset()
        triplet_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        timer.set()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            timer.cnt('rd')
            if (flag):
                with torch.no_grad():
                    writer = SummaryWriter('logs', flush_secs=5)
                    writer.add_graph(model=model, input_to_model=(img, target, target_cam, target_view))
                    flag = False
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss, id_loss, triplet_loss = loss_fn(score, feat, target, target_cam)
                timer.cnt('fw')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            timer.cnt('bw')

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            id_loss_meter.update(id_loss.item(), img.shape[0])
            triplet_loss_meter.update(triplet_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID_Loss: {:.3f}, Triplet_Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, Time Info: {}"
                        .format(epoch, (n_iter + 1), len(train_loader),
                                loss_meter.avg, id_loss_meter.avg, triplet_loss_meter.avg, acc_meter.avg,
                                scheduler._get_lr(epoch)[0], timer.show()))
            timer.set()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                full_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_center": optimizer_center.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(full_dict,
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
        # CNN 进行解冻
        if (cfg.SOLVER.FREEZE_PARAM and cfg.SOLVER.FREEZE_EPOCH == epoch):
            for p in model.base.trans_1.parameters():
                p.requires_grad = True
            for i in range(2, 13):
                model_base = model.base
                conv_trans = getattr(model_base, "conv_trans_" + str(i))
                conv_trans = conv_trans.trans_block
                for p in conv_trans.parameters():
                    p.requires_grad = True


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
