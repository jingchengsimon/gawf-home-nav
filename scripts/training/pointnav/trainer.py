"""Formal DD-PPO checkpoint and policy adapters; frozen vendor stays unchanged."""

import json
import os
from pathlib import Path
import time

import torch

from habitat_baselines.rl.ddppo.algo import ddppo_trainer as trainer_module
from habitat_baselines.rl.ddppo.algo.ddppo_trainer import DDPPOTrainer

from scripts.training.pointnav.models import make_gawf_policy


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_formal_policy(model, trainer):
    if model != "gawf2":
        return
    original = trainer_module.PointNavResNetPolicy

    def factory(*args, **kwargs):
        if args:
            raise ValueError("Unexpected positional policy arguments")
        return make_gawf_policy(
            kwargs["observation_space"], kwargs["action_space"],
            trainer.config.RL.PPO, trainer.config.RL.DDPPO,
            kwargs["goal_sensor_uuid"],
        )

    trainer_module.PointNavResNetPolicy = factory
    return original


class FormalDDPPOTrainer(DDPPOTrainer):
    def prepare_formal(self, results, model, seed, target_steps, checkpoint_steps):
        self.formal_target_steps = target_steps
        self.formal_results = Path(results)
        self.formal_checkpoint = self.formal_results / "checkpoints/interrupted.pth"
        self.formal_meta = dict(model=model, seed=seed, target_steps=target_steps,
                                checkpoint_steps=checkpoint_steps)
        previous = None
        if self.formal_checkpoint.exists():
            previous = torch.load(self.formal_checkpoint, map_location="cpu")
            if previous.get("formal_meta") != self.formal_meta:
                raise RuntimeError("Resume checkpoint protocol differs from formal manifest")
            if not {"state_dict", "optim_state", "lr_sched_state", "requeue_stats"} <= set(previous):
                raise RuntimeError("Resume checkpoint lacks complete trainer state")
        self.formal_updates_completed = int(
            previous["requeue_stats"]["start_update"] if previous else 0
        )
        self.formal_last_saved_steps = int(
            previous["requeue_stats"]["count_steps"] if previous else 0
        )
        self.formal_prior_time = float(
            previous["requeue_stats"]["prev_time"] if previous else 0.0
        )
        self.formal_started = time.monotonic()
        self.formal_scheduler = None
        self.formal_frames = self.formal_last_saved_steps
        self.formal_checkpoint_steps = checkpoint_steps

        original_scheduler = trainer_module.LambdaLR

        def capture_scheduler(*args, **kwargs):
            scheduler = original_scheduler(*args, **kwargs)
            self.formal_scheduler = scheduler
            return scheduler

        trainer_module.LambdaLR = capture_scheduler
        original_save = trainer_module.save_interrupted_state

        def save_on_signal(state, filename=None):
            state["formal_meta"] = self.formal_meta
            atomic_torch_save(state, filename or self.formal_checkpoint)

        trainer_module.save_interrupted_state = save_on_signal
        self._formal_restore_hooks = (original_scheduler, original_save)

    def finish_formal(self):
        if hasattr(self, "_formal_restore_hooks"):
            trainer_module.LambdaLR, trainer_module.save_interrupted_state = self._formal_restore_hooks

    def save_checkpoint(self, file_name, extra_state=None):
        # The frozen loop calls this on rank 0 after every update. Its ordinary
        # policy-only checkpoint cannot resume Adam, so write a complete state.
        del file_name
        if extra_state is None or "step" not in extra_state:
            raise RuntimeError("Formal checkpoint requires measured global frames")
        if self.formal_scheduler is None:
            raise RuntimeError("Formal LR scheduler was not captured")
        frames = int(extra_state["step"])
        self.formal_frames = frames
        checkpoint_due = (
            frames - self.formal_last_saved_steps >= self.formal_checkpoint_steps
            or frames >= self.formal_target_steps
        )
        if checkpoint_due:
            state = dict(
                state_dict=self.agent.state_dict(),
                optim_state=self.agent.optimizer.state_dict(),
                lr_sched_state=self.formal_scheduler.state_dict(),
                config=self.config,
                formal_meta=self.formal_meta,
                requeue_stats=dict(
                    env_time=0.0, pth_time=0.0, count_steps=frames,
                    count_checkpoints=self.formal_updates_completed,
                    start_update=self.formal_updates_completed,
                    prev_time=self.formal_prior_time + time.monotonic() - self.formal_started,
                ),
            )
            atomic_torch_save(state, self.formal_checkpoint)
            self.formal_last_saved_steps = frames
        atomic_json(dict(status="complete" if frames >= self.formal_target_steps else "running",
                         global_frames=frames, updates_completed=self.formal_updates_completed,
                         saved_global_frames=self.formal_last_saved_steps,
                         target_global_frames=self.formal_target_steps,
                         checkpoint_path=str(self.formal_checkpoint)),
                    self.formal_results / "progress.json")
