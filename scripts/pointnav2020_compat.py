"""Strict runtime compatibility patches for the frozen 2020 PointNav stack."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import threading


_SIM_MODULE = "habitat.sims.habitat_simulator.habitat_simulator"
_DDP_MODULE = "habitat_baselines.rl.ddppo.algo.ddp_utils"
_TRAINER_MODULE = "habitat_baselines.rl.ddppo.algo.ddppo_trainer"
_EXPECTED_SOURCES = {
    _SIM_MODULE: (
        "vendor/habitat-lab-2020/habitat/sims/habitat_simulator/"
        "habitat_simulator.py",
        "c2675517bb53d549238fa70e80570a555ae78f17ab761c2e9fb420493c38ac2c",
    ),
    _DDP_MODULE: (
        "vendor/habitat-lab-2020/habitat_baselines/rl/ddppo/algo/ddp_utils.py",
        "eca04dcc3a0451ae207fe1cccf25a735f6c1ba4f8aa51f7b599e133433e30bd1",
    ),
    _TRAINER_MODULE: (
        "vendor/habitat-lab-2020/habitat_baselines/rl/ddppo/algo/ddppo_trainer.py",
        "af0de7fb54bbd3ffb60bd751f99cdceb29566915417c0c114bca254e5b674f83",
    ),
}

_lock = threading.RLock()
_installed_root: Path | None = None
_records: list[dict[str, object]] = []


def _replace_exact(
    source: str,
    old: str,
    new: str,
    expected_count: int,
    description: str,
) -> tuple[str, str]:
    count = source.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"PointNav 2020 compatibility pattern count changed for "
            f"{description}: expected {expected_count}, found {count}"
        )
    return source.replace(old, new), description


def _patch_simulator(source: str) -> tuple[str, list[str]]:
    changes = []
    source, change = _replace_exact(
        source,
        ".scene.id",
        ".scene_id",
        2,
        "map the two legacy scene.id accesses to SimulatorConfiguration.scene_id",
    )
    changes.append(change)
    source, change = _replace_exact(
        source,
        "habitat_sim.SensorSpec()",
        "habitat_sim.CameraSensorSpec()",
        1,
        "construct a visual CameraSensorSpec for RGB-D resolution and HFOV",
    )
    changes.append(change)
    source, change = _replace_exact(
        source,
        'sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)',
        "sim_sensor_cfg.hfov = float(sensor.config.HFOV)",
        1,
        "set the Habitat-Sim 0.2.5 CameraSensorSpec.hfov property directly",
    )
    changes.append(change)

    marker = "        agent_config.sensor_specifications = sensor_specifications\n"
    navmesh_adapter = (
        "        # Match the 2020 LoCoBot embodiment when Sim 0.2.5 loads/rebuilds navmeshes.\n"
        "        if (\n"
        "            abs(float(agent_config.radius) - 0.18) > 1e-9\n"
        "            or abs(float(agent_config.height) - 0.88) > 1e-9\n"
        "        ):\n"
        "            raise RuntimeError(\n"
        "                \"PointNav 2020 requires agent radius=0.18 and height=0.88\"\n"
        "            )\n"
        "        sim_config.navmesh_settings = habitat_sim.nav.NavMeshSettings()\n"
        "        sim_config.navmesh_settings.set_defaults()\n"
        "        sim_config.navmesh_settings.agent_radius = agent_config.radius\n"
        "        sim_config.navmesh_settings.agent_height = agent_config.height\n"
        "\n"
        + marker
    )
    source, change = _replace_exact(
        source,
        marker,
        navmesh_adapter,
        1,
        "configure Sim 0.2.5 navmesh settings from the frozen 0.18m/0.88m agent",
    )
    changes.append(change)
    return source, changes


def _patch_ddp_utils(source: str) -> tuple[str, list[str]]:
    source, change = _replace_exact(
        source,
        'subprocess.check_call(shlex.split("scontrol requeue {SLURM_JOBID}"))',
        'subprocess.check_call(shlex.split(f"scontrol requeue {SLURM_JOBID}"))',
        1,
        "format SLURM_JOBID in the original DD-PPO requeue command",
    )
    return source, [change]


def _patch_ddppo_trainer(source: str) -> tuple[str, list[str]]:
    source, progress_change = _replace_exact(
        source,
        "                count_steps += stats[2].item()\n",
        "                count_steps += stats[2].item()\n"
        "                if getattr(self, 'formal_target_steps', None) is not None:\n"
        "                    self.formal_frames = count_steps\n"
        "                    self.formal_updates_completed = update + 1\n",
        1,
        "expose measured global frame count to every formal rank",
    )
    source, change = _replace_exact(
        source,
        "                        count_checkpoints += 1\n\n            self.envs.close()",
        "                        count_checkpoints += 1\n"
        "                # Formal runs stop by measured global frames, on every rank.\n"
        "                target = getattr(self, 'formal_target_steps', None)\n"
        "                if target is not None and count_steps >= target:\n"
        "                    break\n\n"
        "            self.envs.close()",
        1,
        "stop formal training after the measured global frame target on all ranks",
    )
    return source, [progress_change, change]


_PATCHERS = {
    _SIM_MODULE: _patch_simulator,
    _DDP_MODULE: _patch_ddp_utils,
    _TRAINER_MODULE: _patch_ddppo_trainer,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prepare_patch(
    module_name: str,
    path: Path,
    source_path: str,
    expected_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    original = path.read_bytes()
    actual_sha256 = _sha256(original)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Refusing to patch {module_name}: expected source SHA256 "
            f"{expected_sha256}, found {actual_sha256} at {path}"
        )
    patched_text, modifications = _PATCHERS[module_name](original.decode("utf-8"))
    patched = patched_text.encode("utf-8")
    return patched, {
        "module": module_name,
        "source_path": source_path,
        "original_sha256": actual_sha256,
        "patched_sha256": _sha256(patched),
        "modifications": modifications,
    }


class _PatchedLoader(importlib.abc.Loader):
    def __init__(
        self, module_name: str, path: Path, original_sha256: str, patched: bytes
    ) -> None:
        self.module_name = module_name
        self.path = path
        self.original_sha256 = original_sha256
        self.patched = patched

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        actual_sha256 = _sha256(self.path.read_bytes())
        if actual_sha256 != self.original_sha256:
            raise ImportError(
                f"Refusing to import {self.module_name}: source changed after "
                f"install(); expected {self.original_sha256}, found {actual_sha256}"
            )
        code = compile(self.patched, str(self.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _PatchedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, loaders: dict[str, _PatchedLoader]) -> None:
        self.loaders = loaders

    def find_spec(self, fullname, path=None, target=None):
        loader = self.loaders.get(fullname)
        if loader is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname, loader.path, loader=loader
        )


def install(root: str | Path) -> list[dict[str, object]]:
    """Install strict hooks before any legacy ``habitat`` import.

    Call this at module top level so ``forkserver`` workers install the same
    hooks while rebuilding their interpreter state.
    """

    global _installed_root, _records
    project_root = Path(root).expanduser().resolve()
    legacy_root = project_root / "vendor/habitat-lab-2020"

    with _lock:
        if _installed_root is not None:
            if project_root != _installed_root:
                raise RuntimeError(
                    f"PointNav compatibility already installed for {_installed_root}"
                )
            return patch_records()

        imported = sorted(
            name
            for name in sys.modules
            if name == "habitat"
            or name.startswith("habitat.")
            or name == "habitat_baselines"
            or name.startswith("habitat_baselines.")
        )
        if imported:
            raise RuntimeError(
                "install(root) must run before importing legacy Habitat; already "
                f"loaded: {', '.join(imported[:5])}"
            )
        if not legacy_root.is_dir():
            raise RuntimeError(f"Missing frozen Habitat Lab source: {legacy_root}")

        loaders = {}
        records = []
        for module_name, (relative_path, expected_sha256) in _EXPECTED_SOURCES.items():
            source_path = project_root / relative_path
            patched, record = _prepare_patch(
                module_name, source_path, relative_path, expected_sha256
            )
            loaders[module_name] = _PatchedLoader(
                module_name, source_path, expected_sha256, patched
            )
            records.append(record)

        legacy_path = str(legacy_root)
        sys.path[:] = [entry for entry in sys.path if entry != legacy_path]
        sys.path.insert(0, legacy_path)
        sys.meta_path.insert(0, _PatchedFinder(loaders))
        _records = records
        _installed_root = project_root
        return patch_records()


def patch_records() -> list[dict[str, object]]:
    """Return JSON-serializable provenance for the installed source patches."""

    with _lock:
        if _installed_root is None:
            raise RuntimeError("PointNav 2020 compatibility hook is not installed")
        return [
            {
                **record,
                "modifications": list(record["modifications"]),
            }
            for record in _records
        ]
