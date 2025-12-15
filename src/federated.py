import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="pass in a parameter")

    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    )
    parser.add_argument(
        "--num_corrupt", type=int, default=2, help="number of corrupt agents"
    )
    parser.add_argument(
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.1, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1, help="servers learning rate"
    )
    parser.add_argument(
        "--target_class", type=int, default=7, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.5,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )
    parser.add_argument(
        "--theta", type=int, default=8, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.25,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "median",
            "avg_align",
            "alignins",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "rfa",
            "scopemm",
            "scope",
            "mpsaguard",
            "mpsa",
            "median_guard",
            "median_guard_align",
            "tda_only"
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--align_cluster_method", type=str, default="none", choices=["none", "kmeans", "spectral", "dbscan", "hdbscan", "finch"], help="clustering method for avg_align diagnostics")
    parser.add_argument("--align_cluster_k", type=int, default=2, help="number of clusters to use when align_cluster_method is kmeans")
    parser.add_argument("--align_spectral_clusters", type=int, default=2, help="number of clusters for spectral clustering in avg_align")
    parser.add_argument("--align_dbscan_eps", type=float, default=0.3, help="DBSCAN epsilon for avg_align clustering")
    parser.add_argument("--align_dbscan_min_samples", type=int, default=2, help="DBSCAN min_samples for avg_align clustering")
    parser.add_argument("--align_hdbscan_min_cluster_size", type=int, default=2, help="HDBSCAN min_cluster_size for avg_align clustering")
    parser.add_argument("--align_hdbscan_min_samples", type=int, default=1, help="HDBSCAN min_samples for avg_align clustering")
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--pretrained_path", type=str, default=None, help="path to pretrained model checkpoint for ResNet9 (CIFAR-10) or VGG9 (CIFAR-100)")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--alignins_layer_norm", action="store_true", default=False, help="enable layer-wise normalization for AlignIns aggregation")
    parser.add_argument("--tda_use_median_anchor", action="store_true", default=False, help="use median aggregation as TDA anchor (False=use global model, True=use median)")
    parser.add_argument("--avg_align_topk_ratio", type=float, default=0.3, help="top-k ratio used in avg_align sign-alignment (default 0.3)")
    parser.add_argument("--median_guard_two_sided", action="store_true", default=False, help="use two-sided MZ filtering in median_guard (filter too high or too low)")
    
    # ScopeMM aggregation method parameters
    parser.add_argument("--eps", type=float, default=1e-12, help="epsilon for numerical stability in ScopeMM")
    parser.add_argument("--percent_select", type=float, default=20.0, help="percentage of clients to select per wave expansion round in ScopeMM")
    parser.add_argument("--combine_method", type=str, default="max", choices=["max", "euclidean", "mahalanobis", "fedid_dynamic", "scope"], help="method to combine multiple metrics in ScopeMM")
    parser.add_argument("--use_candidate_seed", action="store_true", default=False, help="use candidate seed selection in ScopeMM wave expansion")
    parser.add_argument("--use_mpsa_prefilter", action="store_true", default=False, help="use MPSA prefilter in ScopeMM/Scope")
    parser.add_argument("--candidate_seed_ratio", type=float, default=0.5, help="ratio of candidates for seed selection in ScopeMM")
    parser.add_argument("--fedid_reg", type=float, default=1e-3, help="regularization coefficient for FedID dynamic weighting in ScopeMM")
    parser.add_argument("--use_norm_prefilter", action="store_true", default=False, help="enable norm-based prefilter before MPSA in Scope aggregation")
    parser.add_argument("--norm_prefilter_lower", type=float, default=0.4, help="lower multiplier (relative to median norm) for Scope norm prefilter")
    parser.add_argument("--norm_prefilter_upper", type=float, default=3.0, help="upper multiplier (relative to median norm) for Scope norm prefilter")

    args = parser.parse_args()

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    per_data_dict = {
        "rounds": {"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50},
        "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
    }

    args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)

    train_dataset, val_dataset = utils.get_datasets(args.data)
    backdoor_train_dataset = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

    if args.data != "tinyimagenet":
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(
            copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
        )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(
            poisoned_val_set_only_x.dataset,
            args,
            idxs,
            poison_all=True,
            modify_label=False,
        )
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(
            copy.deepcopy(val_dataset),
            idxs,
            runtime_poison=True,
            args=args,
            modify_label=False,
        )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # initialize a model, and the agents
    global_model = models.get_model(args.data, args).to(args.device)

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)

    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                )
        else:
            agent = Agent(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
            )
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )

    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)
        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = agents[agent_id].local_train(
                    global_model,
                    criterion,
                    rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                )
            else:
                update = agents[agent_id].local_train(
                    global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask
                )
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params
        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
            global_model, agent_updates_dict
        )

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    num_classes=args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                del test_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Backdoor ACC:           %.4f" % poison_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc

        logging.info("------------------------------".format(rnd))

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")
    
    # Save checkpoint at the end of training with parameter information in filename
    checkpoint_name_parts = [
        'checkpoint',
        f'data_{args.data}',
        f'attack_{args.attack}',
        f'defense_{args.aggr}',
        f'corrupt_{args.num_corrupt}',
        f'poison_{args.poison_frac:.2f}',
        f'agents_{args.num_agents}',
        f'rounds_{args.rounds}',
    ]
    
    if args.non_iid:
        checkpoint_name_parts.append(f'noniid_beta_{args.beta:.2f}')
    else:
        checkpoint_name_parts.append('iid')
    
    if args.exp_name_extra:
        checkpoint_name_parts.append(args.exp_name_extra)
    
    checkpoint_filename = '_'.join(checkpoint_name_parts) + '.pth'
    checkpoint_path = os.path.join(args.log_dir, checkpoint_filename)
    
    checkpoint = {
        'model_state_dict': global_model.state_dict(),
        'best_acc': best_acc,
        'best_asr': best_asr,
        'best_bcdr_acc': best_bcdr_acc,
        'rounds': args.rounds,
        'args': args,
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info("Checkpoint saved to: %s" % checkpoint_path)