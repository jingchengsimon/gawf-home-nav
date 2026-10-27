# Gibson PointNav v2

## Current comparison: RGB-only + fixed initial goal

The current lightweight comparison uses Gibson Habitat scenes and **PointNav v2
episodes**, with the Habitat Challenge 2020 DD-PPO training code plus the recorded
compatibility adapter. The older RGB-D and online GPS/compass protocols are
historical experiments, not interchangeable results.

| Setting | Current RGB-only protocol |
| --- | --- |
| Policy observations | `rgb`, fixed initial-frame `pointgoal = g0` |
| Internal history | Previous self-issued action and recurrent state |
| Excluded policy inputs | Depth, online GPS/compass, true pose, updated goal, maps, hidden simulator metadata |
| Visual encoder | Trainable ResNet18, 256×144 RGB |
| Recurrent comparison | Two-layer LSTM512 versus parameter-matched two-layer GaWF |
| Training | seed0 per model; target 1M, actual 1,000,064 global frames |
| Sampling | 2 environments/GPU, 64-step rollouts |
| Episode limit / success radius | 500 actions / 0.36 m |

GaWF feedback uses the previous detached clean recurrent hidden state; RoboTHOR's
policy-logit feedback is a separate adapter. The initial goal cue repeats unchanged
throughout an episode. `CheckedRGBPointNav2020` checks that observation keys are
exactly `rgb` and `pointgoal` and that `g0` stays fixed. Evaluators can read true
state for metrics and visualization. The original distance-progress reward uses
simulator truth as training supervision; reward/info do not enter the policy.

The authoritative files are [protocol.json](../configs/pointnav2020/light_rgb/protocol.json),
[task.yaml](../configs/pointnav2020/light_rgb/task.yaml) and
[ddppo.yaml](../configs/pointnav2020/light_rgb/ddppo.yaml).
`DATASET.TYPE=PointNav-v1` names the loader; it does not change the episode dataset
from Gibson v2 to v1. Gibson scenes require the applicable dataset license.

### Scene split

- Train: Convoy, Eagerville, Haxtun, Nuevo, Reyno, Sawpit, Seward, Sumas.
- Val: Cantwell, Eastville, Edgemere, Scioto; 71 episodes per scene, 284 total.

The upstream full episode collection contains 72 train scenes and 14 Val scenes.
The lightweight experiment uses the named subset above, not the full benchmark.

### Completed validation

Both RGB-only training runs reached 1,000,064 frames. Their seed0 evaluations
completed all 284 identical Val episodes, using sampled actions and the same task
settings, with normal exit receipts.

| Model | Successes | SR (%) | SPL (%) | SoftSPL (%) |
| --- | ---: | ---: | ---: | ---: |
| LSTM2 | 0 / 284 | 0.0000 | 0.0000 | 9.1353 |
| GaWF2 | 1 / 284 | 0.3521 | 0.0259 | 4.0604 |

These low-budget, single-seed results do not establish a robust advantage.
Do not substitute the older RGB-D scores. Source IDs, checkpoint and result hashes
are in the [verified snapshot](../project-page/sources/pointnav_validation.json).
The [training/evaluation audit](../project-page/sources/training_budget_audit.json)
found no actor configuration mismatch; reliable task learning was not achieved
under this budget.

## Entry points

| Purpose | File |
| --- | --- |
| Training driver | [run.py](../scripts/training/pointnav/run.py) |
| Training loop / complete checkpoint | [train.py](../scripts/training/pointnav/train.py), [trainer.py](../scripts/training/pointnav/trainer.py) |
| RGB-only config / compatibility | [runtime.py](../scripts/training/pointnav/runtime.py), [compat.py](../scripts/training/pointnav/compat.py) |
| GaWF recurrent adapter | [models.py](../scripts/training/pointnav/models.py) |
| Evaluation | [evaluate.py](../scripts/analysis/pointnav/evaluate.py) |
| Numeric validation audit | [metrics.py](../scripts/analysis/pointnav/metrics.py) |
| Validation / training figures | [plots.py](../scripts/analysis/pointnav/plots.py) |

Run from the repository root with `python -B -m scripts.training.pointnav.run <training-id>`
or `python -B -m scripts.analysis.pointnav.evaluate <evaluation-id>` in the prepared runtime.
The previous RGB-only longrun wrapper has been folded into the runtime and the
single complete-state training loop; the DD-PPO optimization and frame budget are unchanged.

Training/evaluation drivers expect the independent PointNav environment, licensed
data, a matching manifest, unchanged source hashes and machine-specific paths.
The existing RGB-only launcher is restricted to the recorded seed0 protocol; a
new experiment needs its own manifest and output leaf. Current wrappers contain
SJC execution paths, so these links identify the implementation rather than promise
an unconfigured checkout can run immediately. Simulation runs use the project’s
execution host; this documentation update does not launch any runs.

The plotting scripts read verified local result summaries. Source/dependency versions
are pinned in [vendor/sources.json](../vendor/sources.json); the training source is
`habitat-lab-2020` commit `959bd45431edd8024832a877bdc8218015d97a7e`.

## Preserved historical protocols

- `configs/pointnav2020/light/`: legacy lightweight RGB-D + fixed `g0`.
- `configs/pointnav2020/`: historical full-size 2020 RGB-D configurations.
- Original depth + online GPS/compass inference results and manifests remain
  available for provenance. Their retired executable code is recoverable from Git history.

Historical executable code has been retired from `scripts/`; see [the current layout](CODE_LAYOUT.md). The dated notes below preserve prior
protocols, experiments and operational observations. Queue states, permissions,
GPU availability and words such as “current” within those notes describe their
original dates, not live state.

<details>
<summary>Historical experiment notes (2026-10-19 to 2026-10-22)</summary>

# 当前：2020 PointNav × DD-PPO

## 2026-10-22：独立 RGB-only 轻量协议

用户要求暂停现有 Gibson v2 轻量版 RGB-D seed1，并改用 **RGB-only + 固定初始目标 `g0`** 做新的 seed0 对照。旧 seed1 的 LSTM2、GaWF2 进程均已退出，完整状态 checkpoint（模型、Adam、调度器、恢复计数）分别保存于 572,672 和 486,656 实测全局帧；SHA256、独立结果路径与暂停回执见 `experiments/deployment/sjc-pointnav-light-seed1-pause-20261022.json`。旧 seed0、seed1 数据和配置均未覆盖，暂停的 checkpoint 保留可续跑。

新协议文件为 `configs/pointnav2020/light_rgb/{task.yaml,ddppo.yaml,protocol.json}`，训练入口为 `scripts/train_pointnav2020_light_rgb_longrun.py` 和 `scripts/run_pointnav2020_light_rgb.py`。复用获准的 Gibson 场景与 PointNav v2 episodes、原轻量版的八个训练场景、256×144 RGB、500 步上限、动作噪声、两层 recurrent、ResNet18、PPO 与每 GPU 两环境和 64 步 rollout；仅从视觉配置中移除 Depth。仿真器只启用 `RGB_SENSOR`，任务只启用初始坐标系的 `POINTGOAL_SENSOR`；环境 reset/step 与 policy forward 均要求观测键**恰为 `rgb` 和 `pointgoal`**，并逐步检查 `g0` 不变。上一动作及递归状态来自 Agent 自身。无在线 GPS/Compass、真实位姿、更新目标、地图或隐藏 metadata 输入策略。训练奖励仍沿用官方真值距离进展计算，但 reward/info 不进入策略或 GaWF feedback；因此这里只能称为**执行时无特权输入**，不能称训练全程不用仿真真值。

SJC 的 LSTM2、GaWF2 seed0 各 256 帧功能 smoke 均完成两次 PPO 更新，退出码 0，`rank_0.started.json` 记录 `policy_observations=["pointgoal","rgb"]`，完整模型/优化器/调度器 checkpoint 的 hash 与 summary 一致。随后登记两项从零开始、各 1M 实测全局帧的独立训练：`nav-pointnav2020-gibsonv2-rgb-light-lstm2-formal1m-sjc-v1-seed0-20261022` 与 `nav-pointnav2020-gibsonv2-rgb-light-gawf2-formal1m-sjc-v1-seed0-20261022`。它们不从 RGB-D checkpoint 续训，结果分别写入独立的 `results/<ID>/` 和 `logs/<ID>/`；部署回执见 `experiments/deployment/sjc-pointnav-rgb-light-seed0-20261022.json`。提交时仍需以实际 `progress.json`、checkpoint 和完成回执判断进度，不能把 tmux 会话或 `running` 登记当作训练成功。两模型完成后应在同一批四个 Gibson v2 val 场景上重新验证；旧 RGB-D seed0 验证分数不能用于 RGB-only 对照。

2026-10-19 用户选定 Habitat Challenge **2020 PointNav** 作为 Amarel 训练主线，替代最初 depth + GPS/Compass 版本。Agent 每步接收当前 RGB-D、固定初始坐标系目标 g0、上一动作和 LSTM 状态。g0 始终是相对 episode 初始位姿的向量，不随移动更新；不要求模型记住 cue，但要求估计自身运动与空间状态。actor/critic 没有在线位姿、GPS/Compass、目标真值进展、地图或其他 oracle 观测。

## 可执行配置与来源

- `configs/pointnav2020/task.yaml`：逐项沿用官方 `vendor/habitat-challenge-2020/configs/challenge_pointnav2020_v2.local.rgbd.yaml`；仅将 split 改为 train、路径改为本项目 data 路径。
- `configs/pointnav2020/ddppo.yaml`：沿用官方 `configs/ddppo_pointnav.yaml`；仅改变 task 路径与实验输出路径模板。ResNet50 + LSTM512 ×2、4 env/GPU、64 rollout steps、PPO 两轮、lr=2.5e-4、GLOO 均未改动。SJC 两卡 smoke 已测得与 EmbCLIP 同机并行时的短时吞吐，不能据此推断 Amarel 正式训练吞吐。
- `configs/pointnav2020/protocol.json`：源版本、观测边界、数据要求和未完成项。训练源码为 `vendor/habitat-lab-2020` commit `959bd45431edd8024832a877bdc8218015d97a7e`。不把现代 v0.2.5 trainer 静默替代该源码。
- `scripts/check_pointnav2020.py`：只读核对配置与官方源差异；`--data` 额外核对完整 train/val episode 引用的 GLB 和场景隔离。它不下载、训练或提交作业。需要 PyYAML，无需 GPU/Habitat。

保留官方任务的 640×360 RGB-D、70° HFOV、0.88m 相机、0.18m 半径、500 步上限、0.36m STOP 成功半径、PyRobot LoCoBot Proportional 噪声系数 0.5、Gaussian RGB/Redwood Depth 噪声、禁止碰撞滑行。此次采用官方 **v2 task config 的 -20° 相机 pitch**；此前公开场景 demo 使用旧版 0° 参数，不能把旧 demo 当作本配置的完整训练验收。

训练奖励沿用 legacy NavRLEnv：`-0.001 + d_prev - d_current + 2.5 * SPL_current`，其中默认 `SUCCESS_MEASURE=spl`，SPL 仅成功 STOP 时非零。距离由 simulator 真值计算，是训练监督；reward、info、距离不作为 policy/critic/GaWF 的在线输入。不能声称训练全程不用仿真真值。

`DATASET.TYPE=PointNav-v1` 是官方 Python 加载器名称，**数据包必须是 PointNav Gibson v2**，二者不要混淆。下载清单、落盘目录和权限见 `docs/AMAREL.md` (local deployment notes)。

## 当前完成范围

配置已选择。2026-10-20 用户确定正式训练为两层 LSTM512 与参数匹配的两层 GaWF，各 3 seeds、每 seed 单节点 4 GPU、目标 10M 实际全局帧；六项独立提交 Amarel，walltime 72 小时。配置模板的 `NUM_UPDATES: 10000` 仍是官方模板值，正式入口使用较高 update ceiling 并按实测全局帧停止。输出模板包含 `REQUIRES_EXPERIMENT_ID`，实际运行使用已注册 ID 的独立 paths，不复用历史结果目录。原四卡和两卡 Amarel smoke 取消记录保留；所有新 smoke 仅在 SJC。

训练环境 `.envs/pointnav2020` 现已安装，并通过 Amarel RTX3090 单卡功能预检（62734155）：原 2020 Habitat Lab + Sim0.2.5 兼容层、严格 checkpoint 加载、RGB-D 噪声、固定 g0、正确 navmesh 尺寸均验证通过。原 trainer 的四卡排队任务 62734170 已确认取消。首个两卡 smoke `62734442` 于2026-10-19运行46秒后失败，0/4 updates、0 global frames；`torch.distributed.launch`与原2020 DD-PPO trainer各自建立TCPStore却共用`MASTER_PORT`，第二次绑定报地址占用。其日志和结果leaf保留。现已在外层启动器及训练adapter为两套TCPStore分配不同本机端口，未修改vendor、任务、模型、奖励或PPO参数；修复后的独立两卡 smoke `62755288` 于2026-10-20 09:58 EDT提交，但按用户明确要求在排队时取消，Slurm 记录 `CANCELLED`、运行 0 秒、0/4 updates。所有后续 smoke 在 SJC；Amarel 只用于正式训练。

SJC 已部署 v2 train/val episodes 及其引用的 86 个 Gibson 场景与 navmesh；校验为 train 72 scenes / 3,599,996 episodes、val 14 scenes / 994 episodes。SJC 功能预检 `nav-pointnav2020-gibsonv2-preflight-sjc-v1-seed0-20261020` 通过。两卡 LSTM 训练 smoke `nav-pointnav2020-gibsonv2-ddppo-lstm2-smoke2gpu-v3-seed0-20261020` 完成 4/4 updates、2048 global frames；原 trainer 的 policy-only checkpoint 不足以续训 Adam。新增正式入口在 SJC 的 LSTM 和 GaWF 各完成 256 帧 PPO 更新与完整状态保存；两卡 GaWF `nav-pointnav2020-gibsonv2-ddppo-gawf2-formal2gpu-smoke-v1-seed0-20261020` 完成 1 update / 512 全局帧，两个 rank 正常退出，均记录约 4.21 GB 最大显存并保存完整状态。GaWF 恢复 smoke 从 512 帧的完整状态（含 Adam）继续到 768 帧、共 3 updates。原版可重入 activation checkpoint 与 DDP 的失败尝试保留；通过版本改用 Torch1.13 的非可重入 checkpoint。两层 recurrent 参数量：LSTM 4,333,568，GaWF 4,296,704，差 0.85%。

正式入口为 `scripts/run_pointnav2020_formal.py`、`train_pointnav2020.py --formal` 和 `pointnav2020_formal_job.sh`。GaWF 复用 `scripts/embclip/canonical_core/gawf.py`，反馈仅来自上一步经 reset mask 处理的两层 clean hidden state（detach），不使用 reward、info 或模拟器真值。训练配置保留官方 encoder、动作头、critic、PPO、4 env/GPU 和 64-step rollout。完整模型/优化器/调度器/update 状态约每 10,000 实际全局帧原子替换一次，正常达到 10M 帧时再保存；单次作业提前 10 分钟向所有 rank 发 USR1，在 update 边界保存并请求 Slurm requeue。六项 Slurm `Requeue=1`、`Dependency=(null)` 已回读；实际超时重排尚未发生，不能将 SJC 的中断恢复检查当作 72 小时 Slurm 事件验收。

2026-10-20 提交的正式作业：GaWF seeds 0/1/2 为 `62757476`–`62757478`，LSTM seeds 0/1/2 为 `62757479`–`62757481`；提交回读均为 `PENDING/Priority`，各请求 4 GPU、24 CPU、120 GiB、72 小时。作业彼此无依赖，但实际并发取决于 Amarel 分配。完整身份和结果路径见 `experiments/monitoring/jobs/` 下对应的六份 manifest。

## 独立的 Habitat 轻量版试验

2026-10-20 新增 `configs/pointnav2020/light/` 与 `scripts/train_pointnav2020_light.py`，并将这四个文件按 SHA256 对齐部署到 Amarel；原有六项正式作业及其配置未改动。轻量版仍使用 Gibson v2 PointNav 的仿真器、合法 RGB-D 与固定初始目标 g0、官方 v2 动作/噪声/奖励和 500 步上限。它缩小图像到 256×144（原 640×360，像素数为 16%）、将 ResNet50 换成 ResNet18、每 GPU 环境数从 4 降为 2，并固定训练 8 个 Gibson 场景、验证 4 个场景。LSTM512×2 与参数匹配的 GaWF×2 共用这些改动和原 PPO 超参。此协议是独立的快速比较，不是 Challenge 2020 全量结果；`protocol.json` 的 `smoke_pending` 是运行前冻结的源码字段，实际验收状态以实验 manifest 为准。1M 全局帧/seed 仅为待吞吐和学习曲线验证的试验预算，不构成已提交的 Amarel 长训练。

两个单卡训练 smoke 均在 SJC 完成：LSTM 用 GPU0，GaWF 用 GPU1，各 2 updates / 256 全局帧，退出码 0，完整 checkpoint 含优化器状态且模型张量有限。LSTM checkpoint SHA256 为 `d7af32df...588728893249`，GaWF 为 `9ab3896d...a0856d5c`；完整路径和哈希见对应 manifest。两次启动到退出分别用 51.61 和 51.74 秒，包含初始化，不能作为稳态吞吐或模型效果证据。总参数 LSTM 8,481,125、GaWF 8,444,261，差 0.43%。轻量版尚未完成正式训练、验证集评测或长期吞吐测量；Amarel 没有执行任何轻量 smoke。

2026-10-20 用户指定在 SJC 两卡各跑一个 seed0 轻量任务，目标每模型 1M **实测全局环境帧**。独立入口 `scripts/train_pointnav2020_light_longrun.py` 和 `scripts/run_pointnav2020_light_train.py` 使用相同 8 场景、256×144 RGB-D、ResNet18、两层 recurrent、2 env/GPU、64 帧 rollout 与原 PPO 超参；LSTM 在 GPU0，GaWF 在 GPU1。每约 10,000 全局帧原子替换完整 checkpoint（模型、Adam、调度器与 update/帧数）。长训练的 `NUM_UPDATES` 只是高于目标的安全上限；真正终点由实测帧数触发。每次重启保留独立 attempt 日志，同一 ID 从完整 checkpoint 恢复；达到 1M 后记录 checkpoint SHA256 和 summary。

要在 1M 完成后继续增加预算，创建新的 SJC experiment ID 与 manifest，保留 `model`、`seed`、场景及 source SHA、checkpoint 间隔，设置更高的 `budget.target_global_frames`，并用 `resume_from: {"experiment_id": "<1M ID>", "checkpoint_sha256": "<1M summary 的值>"}` 指向原任务。`run_pointnav2020_light_train.py` 会验证原任务已完成、协议/哈希一致，再将完整状态复制到新结果 leaf，仅在副本中更新目标帧数；原 1M checkpoint、日志和 summary 不覆盖。首次延长尚须在 1M 完成后实际验收，不能把可续跑实现描述为已完成续跑。

SJC 运行 ID 为 `nav-pointnav2020-gibsonv2-light-lstm2-formal1m-sjc-v1-seed0-20261020` 和 `nav-pointnav2020-gibsonv2-light-gawf2-formal1m-sjc-v1-seed0-20261020`；每项的 `results/<ID>/config_snapshot/` 保存源 YAML、协议、已解析 Habitat 配置及启动时 manifest。2026-10-20 19:51:49 UTC 回读两项均已进入 PPO 更新，分别为 640 和 256 全局帧，尚无 10k 帧 checkpoint。用户要求暂停的 SJC EmbCLIP GRU3/GaWF3 任务已退出，最后保存的各自 checkpoint 都是 5,007,360 帧；两任务的更高日志进度不是可续跑状态。暂停和新任务证据见 `experiments/deployment/sjc-pointnav2020-light1m-20261020.json`。

2026-10-20 SJC GPU0 顺序完成轻量 PointNav LSTM2 单 seed 的三组 2,048 帧吞吐短测，其他五个 1M 任务保持运行。除环境数与 rollout 长度外，沿用相同 Gibson v2 八场景、256×144 RGB-D、ResNet18、PPO 两轮/两 mini-batch 与 seed0；每组仅在首轮更新保存一次 `ckpt.0.pth`，计时区间剔除首轮及其写盘时间。剔除首轮更新和启动时间后，`2×64` 为 5.90 全局帧/秒，`4×64` 为 6.04（+2.28%），`2×128` 为 5.75（−2.52%）；环境步进占各稳态区间约 92.7%、96.2%、93.6%。三个任务均正常退出且完成 2,048 帧，原始日志、配置快照和完整结果在各自 `results/<ID>/` 与 `logs/<ID>/`，汇总见 `experiments/deployment/sjc-pointnav2020-light-throughput-20261020.json`。每组仅一次短测且共享 GPU/主机负载，2.28% 差异不宜视为可靠提速；当前两项 1M 对照继续沿用 2×64。

## 统一进度登记

2026-10-21 10:06 EDT 更新了 SJC 轻量训练 SR/SPL 快照：`results/navigation-training-metrics-light5m-20261021/pointnav_training_sr_spl.{png,pdf}`，精确标量及来源在同目录 `metrics_snapshot.json`。两条曲线分别为 LSTM2 与 GaWF2 seed0，截取到 727,296 与 653,568 全局帧；它们是训练中完成 episode 的 rolling-50-PPO-update 指标，不是验证集 SR/SPL，也不是 1M 终点结果。

2026-10-21 21:21 EDT 更新 seed0 训练指标快照：`results/navigation-training-metrics-seed0-20261021/pointnav_training_sr_spl.{png,pdf}`，逐点数据及 event 来源见同目录 `metrics_snapshot.json`。LSTM2 曲线到 1,000,064 帧；GaWF2 曲线到 991,360 帧。LSTM2 的运行 manifest 与 summary 已确认完成；此图仍只呈现训练 episode 的 rolling-50-update SR/SPL，不是验证集评测。

2026-10-21 16:30 EDT 用户要求两项 seed0 **都完成后**自动提交 seed1。SJC 已登记 LSTM2/GaWF2 各一项独立 seed1、每项 1M 全局帧：ID 分别为 `nav-pointnav2020-gibsonv2-light-lstm2-formal1m-sjc-v1-seed1-20261021`（GPU0）和 `nav-pointnav2020-gibsonv2-light-gawf2-formal1m-sjc-v1-seed1-20261021`（GPU1）。两项当时均为 `queued`，结果与日志 leaf 尚不存在。`gate-pointnav-light-seed1-20261021` 会话运行 `scripts/queue_pointnav2020_light_seed1.py`：等待两项 seed0 的 manifest、summary、退出回执、实测与保存帧数、完整 checkpoint SHA/状态一致，且旧 tmux 会话及进程退出，再启动两个 seed1；seed1 从头训练，不继承 seed0 权重。16:30 EDT seed0 LSTM2/GaWF2 为 905,856/832,256 帧，尚未满足门槛；SJC `./progress` 已显示两项 seed1 排队且无未登记导航任务。部署回执为 `experiments/deployment/sjc-pointnav-light-seed1-armed-20261021.json`，触发后的实际启动证据另写 `experiments/deployment/sjc-pointnav-light-seed1-gate-20261021.json`；排队不代表 seed1 已训练。

2026-10-21 22:36 EDT 复查：seed0 两项均已在 1,000,064 帧正常完成，但 tmux 保留了退出码 0 的 dead pane；原 gate 仅用 `tmux has-session`，因而误判仍在运行。已将 `session_alive` 改为检查 pane 是否存活，SJC 对齐部署后重新核验两项 summary、退出回执和完整 checkpoint SHA，重新运行 gate。seed1 LSTM2/GaWF2 均已独立启动并在 `./progress` 登记，分别观测到 512/384 实际全局帧；启动回执在 `experiments/deployment/sjc-pointnav-light-seed1-gate-20261021.json`。当时 GPU0/GPU1 分别已用 44,024/43,835 MiB，利用率约 96%/95%，与同机 EmbCLIP 叠跑，后续吞吐与稳定性需以实际日志和 checkpoint 判断。

2026-10-22 完成 seed0 两项 1,000,064 帧 checkpoint 的轻量版验证：同一 Gibson PointNav v2 val 的 Cantwell、Eastville、Edgemere、Scioto 各 71 episodes，共每模型 284 episodes，随机动作采样，500 步上限、0.36 m 成功半径。LSTM2：SR 1/284 = 0.3521%，SPL 0.0701%，SoftSPL 12.7752%，平均终点距离 5.5392 m；GaWF2：SR 0/284、SPL 0%、SoftSPL 6.6215%，平均终点距离 5.2446 m。两个完成 manifest 与远端 summary 一致；本地精确结果快照、SR/SPL/SoftSPL 图见 `results/pointnav-light-seed0-validation-20261022/`，部署回执见 `experiments/deployment/sjc-pointnav-light-val284-20261022.json`。这是四场景、单 seed、1M 预算的轻量协议验证，不能当作官方 Challenge 2020 全量成绩；SR 差异仅一条成功 episode，不足以作稳健模型优劣结论。

SJC 和 Amarel 的 AIM3 项目根目录 `./progress` 现在调用本项目的 `scripts/progress.py`；本项目根目录也提供 `./progress`。读取器按当前主机扫描 `experiments/monitoring/jobs/*.json`，不再使用写死的 EmbCLIP ID。每次提交前先在**执行主机**部署对应完整 ID 的 manifest，核对 `host`、`units[0].result_path` 和训练目标；提交后同轮写入实际 `execution.job_id`（Slurm）或 `execution.session`（tmux），并运行 `python3 -B scripts/rebuild_index.py`。`./progress` 读取 Slurm/tmux 状态以及结果目录中的 `progress.json`、`summary.json` 或 EmbCLIP 日志，显示目标帧数和已保存进度；若发现有导航作业却没有对应的活跃 manifest，会明确显示 `UNREGISTERED navigation job`。新任务开始运行后回读 `./progress` 确认其 ID 和实测帧数；排队、暂停和完成的任务不显示，历史 manifest 与结果仍保留。此登记不改变 SJC-only smoke、Amarel-only 正式训练的主机分工。

官方依据：[2020 任务定义](https://aihabitat.org/challenge/2020/)、[2020 训练入口及 v2 数据链接](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2020#pointnavobjectnav-baselines-and-dd-ppo-training-starter-code)。

---

下文为历史 GPS/Compass demo 原始记录，仅用于工程溯源，不是当前训练协议。

# 历史：最初 GPS/Compass PointNav DD-PPO 复现

当前任务是复现原始 policy，不修改递归核心。两卡开发目标是运行官方预训练 policy 并保存结构化 episode 指标、执行吞吐和视频；只有不存在可用完整 policy 时才转入官方训练配置的有预算 smoke。

## 官方源码

| 组件 | 仓库 | 固定版本 | Commit |
| --- | --- | --- | --- |
| Habitat Lab 与 Baselines | https://github.com/facebookresearch/habitat-lab | v0.2.5 | 17ec6b3c1d6e0f71d2834e62b52a308b567ad843 |
| Habitat Sim | https://github.com/facebookresearch/habitat-sim | v0.2.5 | c8887c8cf421b6df2d77489d42f4e20488e184eb |

本地源码位于本项目 `vendor/`，远端同名目录位于 `<project-root>/vendor/`。第三方仓库中的数学与任务源码保持原样；任何兼容处理均记录于本项目脚本和运行记录。

## 模型与观测

官方模型为 `gibson-2plus-resnet50.pth`：Depth ResNet50、两层 LSTM、hidden size 512，四个离散动作 STOP、前进、左转、右转。policy 的观测仅为 depth 与 `pointgoal_with_gps_compass`，RGB 仅供演示录像。

发布地址：[完整 policy checkpoint](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth)。模型口径及已发表 SPL 以[官方模型表](https://github.com/facebookresearch/habitat-lab/blob/v0.2.5/habitat-baselines/habitat_baselines/rl/ddppo/README.md)为准；采用 sampled actions。严格加载所有 actor、critic、视觉编码和 LSTM 参数，缺失或多余学习参数均为失败，不允许回退随机权重。

这套官方复现允许 GPS/Compass 与原奖励，不能作为仅视觉协议的成绩。用户尚未提供 Gibson 场景目录时，先使用官方公开 test scenes 与其 PointNav episodes；数据域在每份结果中单独标明，不将其称为 Gibson validation。

## 运行与证据

远端部署在独立 Conda 环境 `.envs/habitat-pointnav`，不改动 `aim3_rnn`。GPU 0 与 GPU 1 使用互不重叠的 episode 子集，分别产生 JSONL、summary 与示例视频。这是并行 inference，不能称为两卡 DD-PPO 训练吞吐。

评测启动前固定完整 experiment ID，并创建 manifest。运行过程中将完整命令、输入文件 SHA256、source commits、版本、退出码及每个 GPU 的结果路径保存到该 ID 的结果目录。完成必须有：strict checkpoint load 记录、所有预期 episode 指标、有限数值、summary 和正常退出状态。

训练 smoke 若需要，另设 ID，并使用 tag 自带 `pointnav/ddppo_pointnav.yaml`，只覆写输出路径、两卡资源和 smoke 总预算，不更改模型或优化参数。必须保存 checkpoint 和结构化训练结果；inference 成功不能替代训练 smoke 验收。

## 初次预检

2026-10-19 通过既有 SSH ControlMaster 复用确认：sjc-remote 有两张 RTX A6000，各 49140 MiB；预检时无 CUDA compute process，`/G` 约 524 GiB 可用。此信息是当时快照，每次启动前重新检查。

## 已完成的试跑

完整 experiment ID：`nav-pointnav-testscenes-ddppo-v025-gibson2plus-depth-lstm2-eval10-seed0-20261019`。

2026-10-19 已在两张 RTX A6000 完成 `attempt01`。采用公开 test-scenes 的 val episodes，按 episode 原始索引的奇偶分成两份，各 5 个；本次选中的 10 个 episode 都来自 Skokloster Castle。没有重叠 episode，两个进程正常退出。

| 指标 | 结果 |
| --- | --- |
| 完成 episodes | 10 / 10 |
| Success | 1.000 |
| SPL | 0.96220058 |
| 平均最终目标距离 | 0.084512 m |
| 平均动作数 | 66.6 |
| 两卡合计动作数 | 666 |
| 每卡记录窗口耗时 | GPU0 7.735 s，GPU1 7.034 s |
| 每卡记录窗口 steps/s | GPU0 47.19，GPU1 42.80 |

计时窗口包含环境初始化、模型构造和加载、动作选择、环境步进及视频编码，不含 Python import 和数据下载。每步 CUDA 同步用于诊断计时，因此上述 steps/s 仅是这次小型 inference 的记录，不是稳态最大推理性能，更不是 DD-PPO 训练吞吐。

完整 checkpoint SHA256 为 `a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad`。179 个 state keys 全部严格匹配，完整 policy 共 12,454,917 个参数。未改写权重张量，仅在加载前移除发布文件的 `actor_critic.` 前缀。官方两份 vendor clone 在本地和远端均无源码修改。

运行脚本为 `scripts/eval_pointnav.py` 和 `scripts/run_eval.sh`，直接调用官方 `PointNavResNetPolicy`，而非重写网络。环境使用 Python 3.9.23、PyTorch 1.13.1+cu117、Habitat Sim/Lab/Baselines 0.2.5；精确依赖见 `logs/environment-freeze.txt`。为兼容官方导入路径采用其要求的 `webdataset==0.1.40`；`faster-fifo` 使用可正常编译的 1.5.2。现有 Atari 环境未修改。

本地与远端结果均为 `results/<完整 experiment ID>/attempt01/`，其中 `summary.json` 为汇总，`gpu0/` 与 `gpu1/` 保存各自的 `episodes.jsonl`、原始配置、完整命令与输入 hash、严格加载记录和两段 RGB 录像。RGB 只用于视频，policy 输入为 depth 和 pointgoal。manifest 为 `experiments/monitoring/jobs/<完整 experiment ID>.json`，已标记 complete，active index 为空。

只读状态命令（在本项目根目录运行）：

```bash
python3 -B scripts/status.py nav-pointnav-testscenes-ddppo-v025-gibson2plus-depth-lstm2-eval10-seed0-20261019 --no-update
```

由于官方完整模型可用，本轮执行了 inference 分支，没有启动训练 smoke 或正式训练，也没有接入 GaWF。尚未得到正式 Gibson validation 成绩。

## 后续训练主机

2026-10-19 用户将后续训练数据与实际训练部署主机指定为 Amarel。本地和 Amarel 使用同名 `gawf-home-nav` 项目及相同相对路径；历史 SJC inference 身份、manifest 和结果不变。数据权限与目录安排见 `docs/AMAREL.md` (local deployment notes)。用户尚未申请 Gibson 许可；本次仅准备代码目录，不启动训练。

用户随后要求优先选择原生仅视觉/本体输入的循环 baseline。当前官方 PointNav 的已完成 inference 保留为工程验证，不能据此默认选定其正式训练。


</details>
