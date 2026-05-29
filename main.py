import os
import time
from datetime import datetime
from matplotlib import pyplot as plt
import sys
import random
import argparse
from net.NTSCC_Hyperior import NTSCC_Hyperprior, EHB_NTSCC
import torch.optim as optim
from utils import *
from data.datasets import get_loader, get_test_loader
from config import config


def train_one_epoch(
    epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger
):
    global global_step
    net.train()
    elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs = [
        AverageMeter() for _ in range(7)
    ]
    metrics = [elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs]
    for batch_idx, input_image in enumerate(train_loader):
        optimizer_G.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        start_time = time.time()
        input_image = input_image.to(device)
        global_step += 1

        if config.ehb_mode:
            from loss.ehb_loss import redundancy_cosine_loss

            mse_loss, cbr_y, x_hat, info = net(input_image)
            loss = mse_loss

            red_loss = torch.tensor(0.0, device=device)
            if config.ehb_use_red_loss:
                red_loss, _ = redundancy_cosine_loss(
                    [info["U1"], info["U2"], info["U3"], info["U4"]]
                )
                loss = loss + config.ehb_lambda_red * red_loss

            # Rate predictor CBR loss
            if info.get("rate_continuous") is not None:
                target_rate = config.ehb_target_cbr * 16 * 16 * 3 * 2
                rate_loss = ((info["rate_continuous"] - target_rate) ** 2).mean()
                loss = loss + config.ehb_lambda_cbr * rate_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer_G.step()

            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(cbr_y.item() if isinstance(cbr_y, torch.Tensor) else cbr_y)
            psnr_jscc = 10 * (torch.log(255.0 * 255.0 / mse_loss) / np.log(10))
            psnr_jsccs.update(psnr_jscc.item())

            if ((global_step - 1) % config.print_step) == 224:
                process = (
                    (global_step % train_loader.__len__())
                    / (train_loader.__len__())
                    * 100.0
                )
                rate_str = ""
                if info.get("rate_continuous") is not None:
                    rc = info["rate_continuous"].detach()
                    rate_str = f"RatePred [{rc.min().item():.1f},{rc.mean().item():.1f},{rc.max().item():.1f}]"
                log = " | ".join(
                    [
                        f" Epoch {epoch:4d}",
                        f"Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]",
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"Time {elapsed.avg:.2f}",
                        f"PSNR {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        rate_str,
                    ]
                )
                logger.info(log)
                for i in metrics:
                    i.clear()
        else:
            (
                mse_loss_ntc,
                bpp_y,
                bpp_z,
                mse_loss_ntscc,
                cbr_y,
                x_hat_ntc,
                x_hat_ntscc,
            ) = net(input_image)

            if config.use_side_info:
                cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
                loss = (
                    mse_loss_ntscc
                    + mse_loss_ntc
                    + config.train_lambda * (bpp_y * config.eta + cbr_z)
                )
                cbrs.update(cbr_y + cbr_z)
            else:
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                loss = ntc_loss + mse_loss_ntscc
                cbrs.update(cbr_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer_G.step()

            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            bppys.update(bpp_y.item())
            bppzs.update(bpp_z.item())

            psnr_jscc = 10 * (torch.log(255.0 * 255.0 / mse_loss_ntscc) / np.log(10))
            psnr_jsccs.update(psnr_jscc.item())
            psnr = 10 * (torch.log(255.0 * 255.0 / mse_loss_ntc) / np.log(10))
            psnrs.update(psnr.item())

            if (global_step % config.print_step) == 0:
                process = (
                    (global_step % train_loader.__len__())
                    / (train_loader.__len__())
                    * 100.0
                )
                log = " | ".join(
                    [
                        f" Epoch {epoch:4d}",
                        f"Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]",
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"Time {elapsed.avg:.2f}",
                        f"PSNR_JSCC {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        f"PSNR_NTC {psnrs.val:.2f} ({psnrs.avg:.2f})",
                        f"Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})",
                        f"Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})",
                    ]
                )
                logger.info(log)
                for i in metrics:
                    i.clear()


def test(net, test_loader, logger):
    with torch.no_grad():
        net.eval()
        elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs = [
            AverageMeter() for _ in range(7)
        ]
        PSNR_list = []
        CBR_list = []
        for batch_idx, input_image in enumerate(test_loader):
            start_time = time.time()
            input_image = input_image.cuda()

            if config.ehb_mode:
                mse_loss, cbr_y, x_hat, info = net(input_image)
                loss = mse_loss
                losses.update(loss.item())
                cbrs.update(cbr_y.item() if isinstance(cbr_y, torch.Tensor) else cbr_y)
                elapsed.update(time.time() - start_time)
                psnr_jscc = CalcuPSNR_int(input_image, x_hat).mean()
                psnr_jsccs.update(psnr_jscc)
                log = " | ".join(
                    [
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"Time {elapsed.val:.2f}",
                        f"PSNR {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                    ]
                )
                # logger.info(log)
                PSNR_list.append(psnr_jscc)
                CBR_list.append(cbr_y)
            else:
                (
                    mse_loss_ntc,
                    bpp_y,
                    bpp_z,
                    mse_loss_ntscc,
                    cbr_y,
                    x_hat_ntc,
                    x_hat_ntscc,
                ) = net(input_image)
                if config.use_side_info:
                    cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
                    ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                    ntscc_loss = mse_loss_ntscc + bpp_y * config.eta + cbr_z
                    loss = ntc_loss + ntscc_loss
                    cbrs.update(cbr_y + cbr_z)
                else:
                    ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                    loss = ntc_loss + mse_loss_ntscc
                    cbrs.update(cbr_y)
                losses.update(loss.item())
                bppys.update(bpp_y)
                bppzs.update(bpp_z)
                elapsed.update(time.time() - start_time)

                psnr_jscc = CalcuPSNR_int(input_image, x_hat_ntscc).mean()
                psnr_jsccs.update(psnr_jscc)
                psnr = CalcuPSNR_int(input_image, x_hat_ntc).mean()
                psnrs.update(psnr)
                log = " | ".join(
                    [
                        f" Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"Time {elapsed.val:.2f}",
                        f"PSNR1 {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        f"PSNR2 {psnrs.val:.2f} ({psnrs.avg:.2f})",
                        f"Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})",
                        f"Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})",
                    ]
                )
                # logger.info(log)
                PSNR_list.append(psnr_jscc)
                CBR_list.append(cbr_y)

    if not config.ehb_mode:
        cbr_sideinfo = (
            np.log2(config.multiple_rate.__len__())
            / (16 * 16 * 3)
            / np.log2(1 + 10 ** (net.channel.chan_param / 10))
        )
    else:
        cbr_sideinfo = 0

    logger.info(
        f" Finish test! Average PSNR={psnr_jsccs.avg:.4f}dB, CBR={cbrs.avg + cbr_sideinfo:.4f}"
    )
    return losses.avg, psnr_jsccs.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training/testing script.")
    parser.add_argument(
        "-p",
        "--phase",
        default="train",  # train
        type=str,
        help="Train or Test",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=3,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=1024, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
        type=str,
        help="Result dir name",
    )
    parser.add_argument(
        "--save_log", action="store_true", default=True, help="Save log to disk"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth",
        type=str,
        help="Path to a checkpoint",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.device = device

    workdir, logger = logger_configuration(
        args.name, phase=args.phase, save_log=args.save_log
    )
    config.logger = logger
    logger.info(config.__dict__)

    if config.ehb_mode:
        net = EHB_NTSCC(config).cuda()
    else:
        net = NTSCC_Hyperprior(config).cuda()
    model_path = args.checkpoint
    load_weights(net, model_path)

    if args.phase == "test":
        test_loader = get_test_loader(config)
        test(net, test_loader, logger)
    elif args.phase == "train":
        train_loader, test_loader = get_loader(config)
        global global_step

        if config.ehb_mode:
            optimizer_G = optim.Adam(net.parameters(), lr=config.lr)
            aux_optimizer = None
        else:
            G_params = set(
                p for n, p in net.named_parameters() if not n.endswith(".quantiles")
            )
            aux_params = set(
                p for n, p in net.named_parameters() if n.endswith(".quantiles")
            )
            optimizer_G = optim.Adam(G_params, lr=config.lr)
            aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer_G, milestones=[4000, 4500], gamma=0.1
        )
        tot_epoch = 5000
        global_step = 0
        best_loss = float("inf")
        best_psnr = 0
        steps_epoch = global_step // train_loader.__len__()
        all_loss = []
        all_psnr = []
        for epoch in range(steps_epoch, tot_epoch):
            logger.info(" ======Current epoch %s ======" % epoch)
            # logger.info(f" Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(
                epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger
            )
            lr_scheduler.step()

            loss, psnr = test(net, test_loader, logger)
            all_loss.append(loss)
            all_psnr.append(psnr)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            best_psnr = max(psnr, best_psnr)
            logger.info(f" Best test psnr: {best_psnr:.3f} dB at epoch {epoch}")
            if is_best:
                save_model(
                    net,
                    save_path=workdir
                    + "/models/EP{}_best_loss.model".format(epoch + 1),
                )
                # test(net, test_loader, logger)

            if (epoch + 1) % 100 == 0:
                save_model(
                    net, save_path=workdir + "/models/EP{}.model".format(epoch + 1)
                )

            plot_test_loss(workdir, all_loss, all_psnr)


def plot_test_loss(workdir, losses, all_psnr):
    plt.figure(figsize=(15, 6), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(all_psnr) + 1), all_psnr, label="PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, "test_loss_psnr.png"))
    plt.close()


if __name__ == "__main__":
    main(sys.argv[1:])
