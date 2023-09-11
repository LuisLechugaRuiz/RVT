from act.act_policy import ACTPolicy
from act.tmp.utils import compute_dict_mean, detach_dict
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pickle
from tqdm import tqdm
from rvt.utils.peract_utils import CAMERAS, DATA_FOLDER
from rvt.tmp_dataset import ACTDataset


def main():
    # From config:
    BATCH_SIZE_TRAIN = 3
    NUM_TRAIN = 100
    NUM_VAL = 25
    NUM_WORKERS = 3
    EPOCHS = 3
    TRAIN_REPLAY_STORAGE_DIR = "replay/replay_train"
    TEST_REPLAY_STORAGE_DIR = "replay/replay_val"
    sample_distribution_mode = "transition_uniform"
    device = "cuda:0"
    tasks = ["close_jar"]  # Just testing from now.
    VAL_ITERATIONS = 100
    TRAINING_ITERATIONS = 2000 # Previously: int(10000 // (BATCH_SIZE_TRAIN / 16))
    CKPT_DIR = DATA_FOLDER + "/act_checkpoint"

    train_dataset = ACTDataset(
        tasks,
        BATCH_SIZE_TRAIN,
        TRAIN_REPLAY_STORAGE_DIR,
        DATA_FOLDER,
        NUM_TRAIN,
        NUM_WORKERS,
        True,
        TRAINING_ITERATIONS,
        CKPT_DIR,
    )

    test_dataset = ACTDataset(
        tasks,
        BATCH_SIZE_TRAIN,
        TEST_REPLAY_STORAGE_DIR,
        DATA_FOLDER,
        NUM_VAL,
        NUM_WORKERS,
        False,
        TRAINING_ITERATIONS,
        CKPT_DIR,
    )

    # fixed parameters # TODO: Get this info from Config, remove script vars.
    state_dim = 7
    lr = 1e-4
    lr_backbone = 1e-5
    backbone = "resnet18"
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    dim_feedforward = 2048
    hidden_dim = 256
    kl_weight = 10
    num_queries = 20

    policy_config = {
        "lr": lr,
        "num_queries": num_queries,
        "kl_weight": kl_weight,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "lr_backbone": lr_backbone,
        "backbone": backbone,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "camera_names": CAMERAS,
    }
    config = {
        "num_epochs": EPOCHS,
        "state_dim": state_dim,
        "lr": lr,
        "policy_config": policy_config,
        "task_name": tasks[0],
        "temporal_agg": True,
        "camera_names": CAMERAS,
        "ckpt_dir": CKPT_DIR,
        "seed": 0,
    }
    best_ckpt_info = train_bc(
        train_dataset,
        test_dataset,
        config,
        TRAINING_ITERATIONS,
        VAL_ITERATIONS,
    )
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def train_bc(
    train_dataloader,
    val_dataloader,
    config,
    training_iterations,
    val_iterations,
):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_config = config["policy_config"]

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # TODO: Rethink dataset. The circular buffer is blocking us to access by index.
            data_iter = iter(val_dataloader.dataset.dataset)
            for _ in range(val_iterations):
                raw_data = next(data_iter)
                data = val_dataloader.get_data(raw_data)
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        data_iter = iter(train_dataloader.dataset.dataset)
        for batch_idx in range(training_iterations):
            raw_data = next(data_iter)
            data = train_dataloader.get_data(raw_data)
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            print(f"Batch: {batch_idx}, loss: {loss}")
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def forward_pass(data, policy):
    image_data, qpos_data, target_pose_data, action_data, is_pad = data
    image_data, qpos_data, target_pose_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        target_pose_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, target_pose_data, action_data, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    main()
