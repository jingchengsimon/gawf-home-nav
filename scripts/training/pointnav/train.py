"""Run the original 2020 DD-PPO trainer; add assertions and structured receipts."""

import argparse
import json
import os
import random
import time

from scripts.training.pointnav.runtime import OBSERVATIONS, load_config, run_paths
from scripts.training.pointnav.compat import patch_records

import numpy as np
import torch
import torch.distributed as distributed
from habitat_baselines.rl.ddppo.algo import ddp_utils
from scripts.training.pointnav.trainer import FormalDDPPOTrainer, configure_formal_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--model", choices=("lstm2", "gawf2"), default="lstm2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--formal", action="store_true", required=True)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--checkpoint-steps", type=int)
    args = parser.parse_args()
    assert args.target_steps and args.target_steps > 0
    assert args.checkpoint_steps and args.checkpoint_steps > 0
    assert args.seed >= 0
    results, logs = run_paths(args.experiment_id)
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (results / "checkpoints").mkdir(exist_ok=True)
    ddp_utils.INTERRUPTED_STATE_FILE = str(results / "checkpoints/interrupted.pth")
    config = load_config(args.experiment_id,
                         updates=50000,
                         seed=args.seed, checkpoint_interval=1,
                         log_interval=10)
    torch.set_num_threads(1)
    # Match legacy run_exp and trainer seed handling; no new sensor RNG seeding.
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # torch.distributed.launch holds its own MASTER_PORT until all children exit.
    # The frozen 2020 trainer opens another TCPStore, so it needs a distinct port.
    trainer_port = os.environ["POINTNAV_DDPPO_MASTER_PORT"]
    assert trainer_port != os.environ["MASTER_PORT"]
    os.environ["MASTER_PORT"] = trainer_port
    torch.cuda.set_device(local_rank)
    started = time.monotonic()
    trainer = FormalDDPPOTrainer(config)
    trainer.prepare_formal(results, args.model, args.seed, args.target_steps,
                               args.checkpoint_steps)
    configure_formal_policy(args.model, trainer)
    original_setup = trainer._setup_actor_critic_agent

    def setup(ppo_config):
        original_setup(ppo_config)
        def check_inputs(module, inputs):
            assert set(inputs[0]) == OBSERVATIONS
        trainer.actor_critic.net.register_forward_pre_hook(check_inputs)
        record = {"rank": rank, "local_rank": local_rank,
                  "world_size": distributed.get_world_size(),
                  "gpu": torch.cuda.get_device_name(local_rank),
                  "policy_observations": sorted(OBSERVATIONS),
                  "model": args.model, "seed": args.seed,
                  "parameters": sum(p.numel() for p in trainer.actor_critic.parameters()),
                  "recurrent_parameters": sum(p.numel() for p in trainer.actor_critic.net.state_encoder.parameters()),
                  "patches": patch_records()}
        (results / ("rank_%d.started.json" % rank)).write_text(json.dumps(record, indent=2))
    trainer._setup_actor_critic_agent = setup
    try:
        trainer.train()
    finally:
        trainer.finish_formal()
    assert trainer.formal_frames >= args.target_steps, "Stopped before target; resume full state"
    record = {"rank": rank, "status": "complete", "updates": trainer.formal_updates_completed,
              "world_size": distributed.get_world_size(), "elapsed_s": time.monotonic() - started,
              "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(local_rank),
              "global_frames": trainer.formal_frames}
    (results / ("rank_%d.complete.json" % rank)).write_text(json.dumps(record, indent=2))
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
