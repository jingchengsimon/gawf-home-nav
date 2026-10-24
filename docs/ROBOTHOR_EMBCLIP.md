# RoboTHOR EmbCLIP GRU baseline

2026-10-19，EmbCLIP 独立支线。用户停止HM3D-OVON，保留其源码、环境与资源调查，不再下载HM3D或申请Matterport权限。PointNav与历史RGB-D RoboTHOR demo不变。GPU渲染、官方CLIP checkpoint评测及训练功能smoke均在`sjc-remote`执行；Amarel不做EmbCLIP smoke。2026-10-20用户新增三模型正式训练授权，提交详情见下文。

## 第一步 源码与信息边界

官方入口：[EmbCLIP RoboTHOR baseline](https://github.com/allenai/embodied-clip/blob/main/readme_files/baselines_robothor_objectnav.md)。独立源码 `vendor/embclip-allenact` 固定官方 `allenact` 分支 commit `855fe741ddecaaf2492966d52d3f130e58d3c48f`，`.VERSION=0.5.0`。旧 `vendor/allenact@f00445e4` 是0.4.1a0、没有CLIP插件，不能混用；第三方源码不修改。

下表文件路径相对此独立vendor根：

| 项目 | 结论 | 文件与行号 |
| --- | --- | --- |
| Python/Torch/CUDA | 官方基础环境Python3.8，Torch>=1.6且不为1.8.0；CLIP插件要求>=1.7.1；仓库提供CUDA11.1等环境模板，不是唯一固定CUDA版本 | `conda/environment-base.yml:6–42`，`conda/environment-11.1.yml:5–8`，`allenact_plugins/clip_plugin/extra_environment.yml` |
| AI2-THOR | 插件声明>=2.5.3，但ObjectNav源码实际要求>=3.2.0；必须满足实际导入检查 | `allenact_plugins/robothor_plugin/extra_environment.yml`，`projects/objectnav_baselines/experiments/objectnav_thor_base.py:32–39` |
| 策略传感器 | 只有RGB和目标类别；RGB按CLIP统计量归一化，没有Depth/GPS/Compass/位姿/专家动作/碰撞真值输入 | `projects/objectnav_baselines/experiments/robothor/clip/objectnav_robothor_rgb_clipresnet50gru_ddppo.py:25–39` |
| CLIP与目标 | 冻结RN50视觉预处理，去掉attention pooling，输出2048×7×7；目标为32维learned category embedding | `allenact_plugins/clip_plugin/clip_preprocessors.py:14–36,57–64,93–114`，`projects/objectnav_baselines/experiments/clip/mixins.py:70–87` |
| GRU与头 | 单层GRU512；actor/critic共享belief。当前baseline `add_prev_actions=False`，虽然forward接口有prev_actions，模型没有额外使用上一动作 | `allenact_plugins/navigation_plugin/objectnav/models.py:201–268`，`allenact/embodiedai/models/visual_nav_models.py:191–271` |
| forward与状态 | 输入observations、Memory、prev_actions、masks；episode reset mask=0清零hidden；rollout截断保留末端memory/mask | `allenact/algorithms/onpolicy_sync/policy.py:120–126`，`allenact/embodiedai/models/basic_models.py:245–300,332–372,490–573`，`allenact/algorithms/onpolicy_sync/storage.py:525–538` |
| 环境总数 | 全局60，不是每GPU60；按指定GPU列表分配：4卡每卡15；8卡分为8/8/8/8/7/7/7/7 | `projects/objectnav_baselines/experiments/robothor/objectnav_robothor_base.py:21`，`projects/objectnav_baselines/experiments/objectnav_thor_base.py:122–179` |
| Rollout/PPO | rollout128，1minibatch，4轮更新，Adam lr3e-4，gamma0.99，GAE0.95，grad norm0.5，clip0.1，value0.5，entropy0.01，advantage normalization；无辅助损失 | `projects/objectnav_baselines/mixins.py:276–328`，`allenact/algorithms/onpolicy_sync/losses/ppo.py:217` |
| DD-PPO计数 | 名义每rollout最多60×128=7680帧；慢worker可能提前截断，吞吐须以实际全局计数为准 | `allenact/algorithms/onpolicy_sync/engine.py:1417–1475` |
| 官方总预算 | pipeline为300M，checkpoint间隔5M；用户将来目标200M是另一预算，不能静默改scheduler。本次仅限时smoke，保留原训练数学与学习率计划 | `projects/objectnav_baselines/mixins.py:294,318–327` |
| Episode上限 | 官方500步；用户已确认沿用，并记录与RobustNav300步的差异 | `projects/objectnav_baselines/experiments/objectnav_base.py:19–24` |
| 奖励边界 | 原训练奖励使用模拟器真值geodesic shaping，但reward/info不作为actor/critic输入 | `allenact_plugins/robothor_plugin/robothor_tasks.py:321–371` |
| checkpoint恢复 | 保存model、optimizer、pipeline、seed、scheduler；恢复需显式`-c`，不是自动发现latest；SIGTERM/SIGINT只关闭worker，不保证保存 | `allenact/algorithms/onpolicy_sync/engine.py:843–889`，`allenact/algorithms/onpolicy_sync/runner.py:263–298` |

静态策略输入检查通过；仍需运行时打印sensor/preprocessor/model输入、严格加载checkpoint和确认tensor shape。机器人渲染内部metadata仅供环境与evaluator使用。

### 分布式启动

单节点用官方`allenact/main.py`启动，显式指定可见GPU列表，保留全局60环境。框架支持`--distributed_ip_and_port`与`--machine_id`（`allenact/main.py:236–255`），但本baseline本身的machine_params没有处理多节点local_worker_ids，不能在多台机器复制单节点命令。官方教程`docs/tutorials/distributed-objectnav-tutorial.md:118–162`提供分配方法，其“60×节点数”例子不适合本任务的固定全局60环境约束。多节点需仅做设备/进程分配adapter，由Slurm/srun启动；不使用官方dmain中的broad rsync、后台SSH或screen。

## 第二步 渲染优先验证

### 2026-10-19 执行主机调整

用户指定把渲染、评测和训练功能 smoke 全部移至 `sjc-remote`。SJC 使用独立 `vendor/embclip-allenact`、`.envs/embclip-robothor` 和独立 data/assets/results/logs leaf；不能把历史 RGB-D RoboTHOR demo 的旧 AllenAct、数据和权重当成本支线资产。SJC 两张 RTX A6000、Vulkan loader/NVIDIA ICD 和 `/G` 容量已在本轮重查，运行前再次确认无计算进程。SJC 渲染入口为 `scripts/embclip/render_sjc.sh <完整ID> <GPU索引>`，前台执行、使用独立结果 leaf；原 Amarel Slurm 渲染入口和已提交任务的历史记录保留。GPU 选择前需核查计算进程，不能只凭 CUDA_VISIBLE_DEVICES 推断 Vulkan 实际 GPU。

SJC 独立最小环境已装 Python3.8.20、AI2-THOR4.3.0、NumPy1.23.5、Pillow8.4.0；`a634e28...` CloudRendering构建已缓存，执行文件SHA256为 `29d5f337ae9b7d7ce0b1b9fb81fc5e3b24d365b1847a1db492baa2327f252f43`。渲染实验 `nav-robothor-embclip-render-sjc-v1-seed0-20261019` 在 `sjc-Lambda-Vector` 的RTX A6000上完成：FloorPlan_Val1_1、RotateRight成功、300×400×3 RGB非空且动作前后变化，退出码0；本地/远端结果leaf相对路径均为 `results/<完整ID>/`，图像SHA256为 `fef721fa18dd690d99b9d118247ca3e3bbcba8d3f23dbd1d8e11c994c589de72`。已人工检查RGB图像。此项只验机器渲染，不是policy执行或checkpoint评测。

用户随后指定评测与训练 smoke 对齐官方最新 AI2-THOR。SJC 独立环境现为 AI2-THOR **5.0.0**，构建 `f0825767cd50d69f666c7f282e54abfe58f1e917`，完整 AllenAct 依赖已装且 `pip check` 通过。SJC 缺少系统 `vulkaninfo`，已把 Ubuntu 的 `vulkan-tools` 包解压到本支线资产 leaf，不使用 sudo；`nav-robothor-embclip-render-sjc-v3-seed0-20261019` 在 RTX A6000 CloudRendering 真实完成 FloorPlan_Val1_1 的 RotateRight、300×400×3 RGB，退出码0，图像 SHA256 `841acc53f8aef7e7d7fb629e8792307e652745401cafbaee86999b1f76db0e86`，本地结果已人工检查。v1 的 4.3.0 结果及 v2 的缺少 `vulkaninfo` 失败记录均保留。5.0.0 是本次选定的新版本，不等同于历史 91139 构建的像素或物理复现。

Amarel 的三项排队渲染任务按新指令已核实：62744213和62744354各运行约1秒后FAILED，均无渲染summary；62744378仍为PENDING时已精确取消，`sacct`核验为CANCELLED且耗时0秒。失败日志和历史manifest保留。SSH故障已定位为工具沙箱边界：沙箱外Amarel的ControlMaster正常（`Master running`），SJC无ControlMaster但直接SSH成功。后续在允许网络的执行环境中优先复用Amarel现有ControlMaster，SJC使用`ControlMaster=no`与`ControlPath=none`直接连接；不再把沙箱内socket/网络错误当作远端不可用。

截至2026-10-19，Amarel 独立最小环境是 AI2-THOR4.3.0，未用于 EmbCLIP smoke；2026-10-20已为正式训练补齐依赖并升级到5.0.0。SJC 独立环境已升级到 5.0.0 并完成全量依赖安装，环境冻结记录为 `logs/embclip-robothor-setup/pip-freeze-ai2thor5.0.0.txt`。两端的缓存 leaf 各自独立；Amarel 的 assets 链接仍落到 scratch。旧 `.ai2thor` 与 Habitat 环境不变。

官方README的`THOR_IS_HEADLESS=True`与此分支实际接口不同：代码读取构造参数`headless=True`，随后选择CloudRendering及安装包的build commit（`objectnav_thor_base.py:59–105,252–256`）。`THOR_IS_HEADLESS`不是有效配置字段，不能照抄。

官方旧commit `91139c909576f3bf95a187c5b02c6fd455d06b48` 经当前下载器查询Linux64与CloudRendering均报 `ValueError: Invalid commit_id ... no build exists ...`，原始回执保留在`logs/embclip-robothor-setup/simulator-cache*.json`。补查官方S3的HTTP/HTTPS均为403，只能确认当前不可公开下载，不能断言对象不存在。精确旧wheel可下载，但包内 `_builds.COMMIT_ID` 实际为 `2eb3d7eabf532a35ec760fd56efc90dc0c85f342`，并非包版本中的91139；其CloudRendering包可访问（636,134,539字节）。未安装或静默替换。审计回执为 `experiments/deployment/amarel-embclip-legacy-build-audit-20261019.json`。4.3.0只验证过机器渲染能力，不能称旧模拟器像素/物理完全复现；本轮后续评测与训练smoke按用户选择统一使用5.0.0。

CloudRendering依赖用户态Vulkan loader及NVIDIA Vulkan ICD/驱动，无需X显示器。登录节点检测到`libvulkan.so.1`不代表GPU节点通过；由预检记录实际GPU型号、驱动、ICD、动作成功与RGB图像。禁止sudo、系统Xorg和管理员方案。

| 完整实验ID | Slurm | 状态 |
| --- | --- | --- |
| `nav-robothor-embclip-render-v100-v1-seed0-20261019` | 62744212 | gpuc001上脚本开始前FAILED，0:53，耗时0秒，无stdout/stderr；未执行渲染 |
| `nav-robothor-embclip-render-v100-v2-seed0-20261019` | 62744235 | gpuc002同样脚本开始前FAILED，0:53、耗时0秒；无渲染结果 |
| `nav-robothor-embclip-render-v100-v3-seed0-20261019` | 62744378 | 排队期间按新指令取消；sacct为CANCELLED、耗时0秒 |
| `nav-robothor-embclip-render-ada-v1-seed0-20261019` | 62744213 | gpuk013上约1秒后FAILED；无渲染summary |

每项预算1GPU、2CPU、8GiB、12分钟，优先CloudRendering，失败才尝试官方旧headless构建。每次渲染尝试最多240秒，结果与原始错误保存在各完整ID的results/logs leaf。两条允许路径均失败时按用户要求停止，不尝试管理员操作。调度/脚本启动前失败与渲染失败分别记录。

额外 Ampere 能力预检 62744354（完整ID `nav-robothor-embclip-render-ampere-v1-seed0-20261019`）在 gpuk006 上约1秒后 FAILED；日志显示结果目录 `/var/lib/slurm/results` 无写权限，未完成渲染。历史日志与自动审批拒绝记录保留在 manifest 中。

## Amarel 资源与存储

复用HM3D支线调查，并于2026-10-19美东晚间刷新（远端快照2026-10-20 02:38 UTC）。gpu/cgpu分区上限仍72h；V100每节点2卡、40CPU、约192GB RAM，Ada有3卡/32CPU/约512GB及4卡/32CPU/约250GB两类；A100有2卡/64CPU和4卡/32或64CPU等类型。型号对照是集群硬件文档，实际分配后由nvidia-smi确认；不把mixed状态理解为GPU空闲。

GPFS `mmlsquota`本轮实查：home/cache使用85,642,336KiB，软/硬100/110GiB；scratch使用485,267,904KiB（约462.79GiB），软/硬1/2TiB，另有1,318,848KiB in_doubt。大文件与独立环境放scratch对应leaf；home只保留源码和小文件。projects可用fileset/配额仍未确认。scratch无备份及清理政策沿用[官方存储说明](https://sites.google.com/view/cluster-user-guide/amarel)，不把共享空闲空间当个人额度。

此前module调查包括CUDA11.7.1/12.1.0/12.8.1、CMake3.26.5、系统GCC11.5.0；本轮最小渲染环境不需要CUDA toolkit。计算节点能否联网等待预检结果，所有大包先在登录节点缓存。

## 公开资产、评测与训练 smoke

SJC 已准备公开 RoboTHOR ObjectNav episodes（train 108,000 / 60 场景；val 1,800 / 15 场景，场景不重叠）、官方 RN50 权重和 130,091,717 步官方 CLIP+GRU checkpoint；原包、SHA256 与验证回执保存在本支线独立 leaf。历史 RGB-D demo 数据与权重未用于本次资产。为功能评测另从全部15个val场景按episode ID确定性抽出100条，保存在 `data/embclip-robothor/eval-smoke-val100-v1`，其 `receipt.json` 记录每个来源文件的哈希、抽取ID及新文件哈希；原val集不改动。

官方 checkpoint 已在 SJC 的100条val smoke子集上闭环评测，完整ID `nav-robothor-embclip-eval-sjc-val100-v1-seed0-20261019`，官方runner正常退出、结果保存100条唯一episode且与subset receipt的ID集合完全一致，覆盖15个val场景；SR=0.41（41/100）、SPL=0.1917236610、平均episode长231.63。AllenAct结果文件名虽标 `test`，实际配置为 `test_on_validation=true`，输入是独立val子集。该小样本与AI2-THOR5.0.0版本不用于直接对照论文test SR0.47/SPL0.20，也不是完整val结果。

训练smoke从随机GRU与目标/融合层开始，视觉编码器保持官方预训练冻结，保留60环境、128rollout和全部PPO设置，墙钟最多1小时。SJC smoke adapter 仅把单个pipeline stage的结束阈值设为7680环境步，保留原300M步学习率衰减和500万步常规checkpoint间隔；阶段正常结束时由官方runner保存checkpoint，独立checkpoint评测已完成因此不启动自动验证worker。完整ID `nav-robothor-embclip-train-smoke-sjc-v1-seed0-20261019` 已正常退出，两张A6000每卡30环境，完成一次PPO更新及7,680全局环境步，官方日志记录elapsed_time=116秒（含启动和关闭），相当于约66.2全局步/秒，仅作本次短时端到端吞吐，不能外推正式长期吞吐。真实checkpoint总步数7,680，SHA256为 `d06367420b5d84af85bb0d7c967d8fe600177145d27409d34471983a6b132a3b`，包含模型和17项optimizer状态，runner退出码0。实测policy参数3,480,775，其中GRU 3,197,952；冻结CLIP RN50视觉编码器38,316,896参数，CLIP完整模型102,007,137参数，视觉预处理器外置，因此不并入policy参数。

恢复续训完整ID `nav-robothor-embclip-resume-smoke-sjc-v1-seed0-20261019` 以SHA固定的7,680步checkpoint为输入，两名worker均加载该文件，保持60环境及相同PPO设置；官方runner正常退出并保存15,360步checkpoint（SHA256 `a3b903c300f7d9c2e532a71ff70d1c2bd4cc9a4e2dbecf9514c473a636cf6e73`）。新权重的17个张量均与输入不同，optimizer仍有17项状态；阶段增加7,680全局步，日志elapsed_time=105秒。恢复功能已验证，但这两次短时吞吐不能作为正式长期速度。

继续恢复剖面完整ID `nav-robothor-embclip-profile-smoke-sjc-v1-seed0-20261020` 已在SJC两张A6000上正常完成，累计76,800全局步及10个rollout，输出checkpoint SHA256为`8de781bc41f5410b26118ea347453e0e4a8bd7e691a50d22816ac0b5e6d97134`，模型17个张量相对输入checkpoint均改变、所有模型张量有限、optimizer有17项状态。该段日志起止约308秒；连同前两段116+105秒，整条从头开始的smoke累计约529秒，低于1小时。稳定区间取30,720→76,800步的164秒，三段每15,360步各耗55、54、55秒，合计**280.98全局步/秒**（同步等效每卡140.49步/秒，不是独立每卡测速）。稳定区间两卡GPU平均利用率62.2%/68.7%、峰值显存17,416/17,237MiB；整机CPU user/system/idle平均约15.8%/11.9%/72.2%，均来自5秒采样。训练日志在30,720/46,080/61,440/76,800步记录reward 0.0751/0.100/0.111/0.105、success 0.0126/0.0150/0.0155/0.0142、total loss 0.211/0.229/0.259/0.269；只说明数值有限，短时曲线不证明性能提升。

2亿步48h目标需要1157.41总steps/s。下表仅把SJC两卡稳定区间的280.98步/秒按GPU数**理想线性外推**，假设无扩展损失、无额外启动开销及相同硬件；没有在4/8/16卡或Amarel实测：

| 每seed GPU | 理想全局步/秒 | 2亿步理想小时 |
| ---: | ---: | ---: |
| 4 | 561.95 | 98.86 |
| 8 | 1123.90 | 49.43 |
| 16 | 2247.80 | 24.72 |

因此8卡即使理想线性也略超48小时；16卡若能保持至少约51.5%的理想效率才可达48小时。原100卡方案允许最多六项16卡作业同时运行，共96卡；2026-10-20用户后来明确改为九项无依赖，若全部同时获资源需144卡。这是提交前吞吐估算，不能用上述线性外推宣称Amarel吞吐已验证。

## 2026-10-20 三模型正式训练提交

用户现授权按2亿步、每模型3seed方案提交官方单层GRU、三层GRU和三层GaWF；两种三层模型与单层GRU按**完整可训练参数**匹配。三种模型保留同一冻结RN50、RGB与目标类别、官方RoboTHOR任务、500步episode上限、全局60环境、128 rollout及原PPO超参。正式预算由官方300M改为200M，学习率线性衰减终点同步改为200M；初始Adam lr仍为3e-4。三层GaWF使用`aim3_gawf_rnn` canonical core原样字节（SHA256见`configs/embclip/formal-source.sha256`），反馈仅为上一时刻网络自身raw policy logits，逐episode清零并detach；PPO序列内按当前参数重算反馈。门控计算使用反向重计算节省显存，不改核心数学。

| 模型 | 层/宽度 | 可训练参数 | 与单层GRU差异 | SJC功能smoke |
| --- | ---: | ---: | ---: | --- |
| GRU1 | 1×512 | 3,480,775 | 基准 | 原训练及恢复smoke已通过 |
| GRU3 | 3×331 | 3,487,953 | +0.206% | 7,680步、一轮PPO、checkpoint SHA256 `d3da59c6515e2e06766f86730b8d6bd40693cddfd31971bf95ce64a12a8c705e` |
| GaWF3 | 3×653 | 3,488,347 | +0.218% | v1在PPO更新OOM、无checkpoint；保留失败记录。重计算后的v2完成7,680步、一轮PPO、checkpoint SHA256 `236b0f2dfeff0cfc3e9df139e25a918b794bdfbf0af08706b95c7872e3d483f9`，含6个门控权重张量和37项optimizer状态 |

SJC另通过CPU结构检查：GaWF reset、前向重计算等价及门控梯度；四节点全局worker ID为`[0–3]`、`[4–7]`、`[8–11]`、`[12–15]`，对应环境数`[4]×12+[3]×4=60`。这不是实际四节点训练验证。Amarel不执行任何EmbCLIP smoke，实际多节点通信、Ada节点CloudRendering以及48小时吞吐仍未验证。

Amarel独立环境已补齐Torch 1.13.1+cu117、AllenAct依赖及AI2-THOR5.0.0，`pip check`通过；固定vendor commit `855fe741ddecaaf2492966d52d3f130e58d3c48f`在本地和Amarel均为干净快照。公开episodes传输后87个文件树SHA256为`308c92d07de335a9c5567eda7e1af0ae15251a9f258d8326eb7131ad9ac6e1c3`，RN50权重SHA256为`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`；AI2-THOR5.0.0的`f0825767cd50d69f666c7f282e54abfe58f1e917` CloudRendering构建仅在登录节点缓存，未在Amarel启动Unity。来源文件清单为`configs/embclip/formal-source.sha256`，其文件SHA256为`17398b45662fa4def06c3d8611ca6aa9f82f25de08c3d78377afa59a2e590590`。本项目根目录没有Git元数据，不能声称完成了根项目的commit/push。

每项正式作业请求`gpu`分区4个四卡Ada节点，共16GPU、128CPU、每节点200G RAM、48小时，训练200M全局步，正常保存间隔5M。原提交顺序为先提交每模型seed0，seed1/2最初带有同模型seed0的`afterok`依赖。提交后的一次历史`squeue`快照全部为PENDING：

| 模型 | seed0（Priority） | seed1（Dependency） | seed2（Dependency） |
| --- | ---: | ---: | ---: |
| GRU1 | 62746459 | 62746467 | 62746468 |
| GRU3 | 62746463 | 62746469 | 62746470 |
| GaWF3 | 62746465 | 62746471 | 62746472 |

完整身份、预算和结果leaf见`experiments/monitoring/jobs/nav-robothor-embclip-*-formal-200m-v1-seed*-20261020.json`。Slurm接受提交只证明排队；模型性能、训练完成和48小时目标均待实际结果验证。`sbatch --test-only`的临时编号62746435未进入队列，不是训练提交。

2026-10-20 01:07:29 EDT按用户新指令对原六项seed1/2作业`62746467, 62746468, 62746469, 62746470, 62746471, 62746472`执行`scontrol update JobId=<id> Dependency=`，逐项回读均为`Dependency=(null)`；三个seed0本来就无依赖。九项现可独立调度，不等待别的seed。该时刻九项仍全部`PENDING`：seed0三项显示`Priority`，seed1/2六项显示`None`。取消依赖不保证九项同时启动；若全部同时运行将占用144张GPU，超过原100卡方案，实际并行度由Amarel调度与资源决定。各manifest保留原`afterok`值作为历史，同时把当前`execution.dependency`改为`null`。

### Amarel 改为每 seed 8 GPU（2026-10-20）

用户要求将九项仍排队的16卡作业改为每项8卡，并采用Amarel `gpu`分区允许的最长墙钟时间。09:40 EDT核实九个旧任务仍为`PENDING`后，以`scancel --state=PENDING`限定取消范围；`sacct`显示九项均为`CANCELLED`、运行时间`00:00:00`。原结果leaf及作业身份保留在原manifest。新任务使用独立`-8gpu-v2-`结果leaf和`formal8_*`配置：每项2节点×4张Ada GPU、每节点32 CPU与200G内存、72小时、200M全局步、60环境、128 rollout、原PPO参数和模型；九项无依赖，全部同时获资源时请求72张GPU。8卡的环境分配为`(8,8,8,8,7,7,7,7)`，节点worker ID分别为`[0–3]`和`[4–7]`。静态配置检查在Amarel登录节点通过；这不代表多节点运行或72小时内完成200M已验证。

| 模型 | seed0 新作业 | seed1 新作业 | seed2 新作业 |
| --- | ---: | ---: | ---: |
| GRU1 | 62754997 | 62754998 | 62754999 |
| GRU3 | 62755000 | 62755001 | 62755002 |
| GaWF3 | 62755003 | 62755004 | 62755005 |

09:40:07 EDT逐项回读`ReqTRES=gres/gpu=8`、`NumNodes=2-2`、`TimeLimit=3-00:00:00`、`Dependency=(null)`，九项均仍为`PENDING`。完整ID、旧作业对应关系、结果leaf见`experiments/monitoring/jobs/nav-robothor-embclip-*-formal-200m-8gpu-v2-seed*-20261020.json`与`experiments/deployment/amarel-embclip-formal8-20261020.json`。部署预检还发现原Amarel源码缺少被正式配置导入的`scripts/embclip/sjc_config.py`，已按本地SHA256补齐；环境缺少`filelock`，已由已验证的SJC环境复制3.16.1离线包并通过`pip check`。新来源清单`configs/embclip/formal8-source.sha256`包含这份共享配置，清单文件SHA256为`89d62dc348f467e1773cc06136bd372a513e1ded4bd90c7a71d295a46ba47f9b`。Amarel没有执行EmbCLIP smoke。

### SJC 双单卡正式训练（2026-10-20）

用户另授权在Amarel排队期间于SJC分别训练一个GRU3与GaWF3 seed0，均以200M全局步为终点。两个独立实验分别使用物理GPU 0/1、每模型60个RoboTHOR训练环境、128 rollout、同一PPO及200M线性学习率计划；每5M步常规保存checkpoint。SJC项目根无Git元数据，因此本地验证后的明确10个adapter文件以SHA256 `332e1ba2121439380e3cef2ee37a06f18b3e755c7b9c247c0e174fe7ae8ec903`部署在各自结果leaf下，不修改SJC项目源码或既有实验。Amarel对应的旧16卡seed0作业已在排队时取消；新8卡seed0作业仍与SJC独立，不从SJC checkpoint续训。

| 模型 | GPU / tmux会话 | 完整实验ID | 2026-10-20 01:01:55 EDT已验证步数 |
| --- | --- | --- | ---: |
| GRU3 | 0 / `embclip-gru3-200m-20261020` | `nav-robothor-embclip-gru3-formal-200m-sjc-v1-seed0-20261020` | 30,720 |
| GaWF3 | 1 / `embclip-gawf3-200m-20261020` | `nav-robothor-embclip-gawf3-formal-200m-sjc-v1-seed0-20261020` | 15,360 |

两者当时均完成至少一次PPO更新、会话仍在运行且日志无OOM；`nvidia-smi`瞬时已用显存分别为32,063/33,598MiB，不能当峰值。训练尚未达到首个5M步保存点，因此无正式训练checkpoint，也不能称200M完成。日志及启动/退出回执位于各自`results/<完整ID>/`；注册见对应`experiments/monitoring/jobs/<完整ID>.json`。

2026-10-20 01:02:41 EDT以既有Amarel ControlMaster核查对应seed0作业：GRU3 `62746463`与GaWF3 `62746465`均仍为`PENDING (Priority)`，运行时长0。SJC训练与它们是独立作业，未更改Amarel队列或依赖。

2026-10-20 09:49 EDT，SJC的AIM3项目根`./progress`已接入导航项目的两个EmbCLIP正式训练ID，保留AIM3原有监控输出，并从各自`train.log`的完整`TRAIN: ... rollout steps`行读取实时全局步数；同时核对准确tmux会话、`exit.json`和checkpoint。导航项目远端`experiments/monitoring/active_jobs.json`已登记两项。该时刻GRU3为`5,898,240/200,000,000`步（2.95%），已保存`5,007,360`步checkpoint；GaWF3为`3,763,200/200,000,000`步（1.88%），尚未到首次5M保存点；两项会话均在运行。部署与原AIM3 wrapper备份见`experiments/deployment/sjc-embclip-progress-20261020.json`。这是当时的日志快照，不代表200M已完成。

## 独立的 1M 步轻量比较

2026-10-20 用户要求停止 Amarel 旧版 PointNav 六项和 EmbCLIP 九项长训练排队，并改交 EmbCLIP 单层 GRU、参数匹配的三层 GRU 与三层 LSTM，各 seed0 的轻量训练。原十五个 Slurm 作业均在 `PENDING/Priority` 时用 `scancel --state=PENDING` 精确取消；`sacct` 确认全部 `CANCELLED`、运行时间 `00:00:00`。旧 manifest、结果 leaf 和作业历史保留。

轻量版保留完整 RoboTHOR ObjectNav train split、RGB 加目标类别、AI2-THOR 5.0.0、冻结 CLIP RN50 与 224×224 输入、官方 500 步 episode 上限、动作、奖励和 PPO 的 128-step rollout、4 次更新及初始学习率 `3e-4`。每模型的总训练预算由 200M 降为 1M 全局步；全局训练环境由 60 降为 16，单任务资源由双节点 8 GPU 改为单节点 1 GPU、16 CPU、80G 内存、24 小时。最初 v1 把线性退火改为 1M，并**意图**把 checkpoint 间隔改为 100k；后者实际未生效，见本节后续修正记录。三模型共享这些设置；环境数和预算变化使它成为独立协议，不能作为旧 200M 结果的直接续训或官方全预算复现。减少环境数降低并发资源占用，不预先保证更高 steps/s。

SJC 运行环境实测可训练参数量：GRU1-512 为 `3,480,775`，GRU3-331 为 `3,487,953`（+0.206%），LSTM3-272 为 `3,473,335`（−0.214%）。LSTM 沿用 AllenAct 原生 `RNNStateEncoder`，hidden/cell 均在 reset mask 处清零；SJC CPU 的双步前向、反向及 reset 对齐检查通过。源码 `scripts/embclip/light*_config.py`、`light_amarel_job.sh` 的哈希清单在 `configs/embclip/light-source.sha256`；新任务各用独立 ID、结果 leaf 和 manifest，进度由项目根 `./progress` 读取。SJC 的三项真实训练 smoke 验收与 Amarel 新任务提交回执以对应 manifest 和部署记录为准。

SJC 的 GRU1、GRU3、LSTM3 均完成真实一轮 PPO：各 2,048 全局步、退出码 0、最终 checkpoint 已记录 SHA256，日志无 traceback；三个 smoke 的 manifest 均已标记 complete。Amarel 于 2026-10-20 16:53 EDT 提交 GRU1 `62761578`、GRU3 `62761579`、LSTM3 `62761580`，均是独立 seed0 作业。Slurm 回读均为单节点、1 GPU、16 CPU、80G、24 小时、`Dependency=(null)`，`./progress` 显示三项 `PENDING`、0/1M；排队不等于训练开始。配置快照在各自 Amarel `results/<ID>/config_snapshot/`，提交、取消和 smoke 的完整回执为 `experiments/deployment/amarel-embclip-light3-20261020.json`。

2026-10-20 19:06 EDT 用户要求利用 SJC 两张卡叠跑上述三项轻量训练。Amarel 的 `62761578`、`62761579`、`62761580` 在仍为 `PENDING` 时精确取消，`sacct` 均为 `CANCELLED`、`00:00:00`，旧结果 leaf 和提交回执保留。SJC 新建独立的 `formal1m-sjc-v1-seed0-20261020` 实验 ID 和结果目录：GRU1 放 GPU0，GRU3 与 LSTM3 放 GPU1；各自 16 环境、1M 全局步，配置与此前 SJC smoke 及 Amarel 轻量提交一致。三个新 manifest 已在 SJC 的 `./progress` 登记，配置快照位于各自 `results/<ID>/config_snapshot/`。这三项与 GPU0 的 PointNav LSTM2、GPU1 的 PointNav GaWF2 同时运行；迁移及资源观测记录见 `experiments/deployment/sjc-embclip-light3-migration-20261020.json`。`running` 只表示会话存活，完成须以 1M checkpoint 与退出码核验。

2026-10-20 19:13:48 EDT 叠跑核验：EmbCLIP GRU1 `20,480/1M`、GRU3 `10,240/1M`、LSTM3 `10,240/1M`，三项均已有实际训练更新；PointNav LSTM2 `256,000/1M`、GaWF2 `163,968/1M`，继续增长。GPU0/GPU1 显存分别为 `14,044/49,140 MiB`、`23,755/49,140 MiB`；截至该快照三项日志无 OOM。共享 GPU 可能降低吞吐，后续以各自进度和 checkpoint 判断。

2026-10-20 晚间修正：上述 SJC `formal1m-sjc-v1` 三任务中的 `pipeline.save_interval=100000` 是 AllenAct 不读取的额外属性；实际 `current_stage.training_settings.save_interval=5000000`。GRU1/GRU3/LSTM3 分别在 204,800/92,160/174,080 步停止，均无 checkpoint，故不能恢复已跑步数。原 manifest 标记 `cancelled`，旧结果和日志保留。新实验使用 `formal1m-sjc-v2` 三个独立 ID，仍是 seed0、每模型 16 环境、1M 全局步；GPU0 为 GRU1，GPU1 叠跑 GRU3 与 LSTM3。配置通过 `ObjectNavPPOMixin.training_pipeline(save_interval=100000)` 设置 AllenAct 实际读取的值，SJC 动态检查已确认；checkpoint 是跨过每 100k 阈值后的整次 rollout 保存，实际步数会高于整十万。

v2 把 1M 作为本轮**停止点**，保留官方 300M 线性学习率计划及其 scheduler 状态，防止 1M 终点学习率归零而使后续续训无效。这是相对 v1 的明确协议变化，三模型一致。v1 的旧哈希清单保留在 `configs/embclip/light-source.sha256`，v2 清单为 `configs/embclip/light-source-v2.sha256`。`scripts/embclip/run_light_formal_sjc_v2.py <实验ID>` 会读取 manifest 并核对源码和结果快照，优先从本实验最新完整 checkpoint 恢复；首次运行则从零开始。每次尝试日志保存在 `attempts/NNN/`，累计日志保存在 `train.log`；恢复时不加 `--restart_pipeline`，因此加载模型、optimizer、训练 pipeline、seed、scheduler。1M 后若要延长，创建新实验 ID 和结果目录，在新 manifest 里写更大 `steps_budget` 与 `resume_from.checkpoint_path`、`resume_from.sha256`，再运行同一入口。新 checkpoint 的实际保存和续训成功仍须以运行产物核验。

2026-10-20 20:04 EDT，用户明确不再运行先前暂停的 SJC GRU3 与 GaWF3 两个 200M 任务。两会话均已停止，manifest 改为 `cancelled` 并从活动索引及 `./progress` 移除；当时各自保存了 5,007,360 步 checkpoint。后续产物清理见下文。此变更不影响三个 1M EmbCLIP 或两个 1M PointNav 任务。

### 21:13 EDT 起的四模型选型

用户将当前 SJC 轻量比较明确改为 **GRU1、GRU3、GaWF1、GaWF3**，均为 seed0、16 个全局训练环境、128 步 rollout、1M 全局帧和约每 100k 帧保存完整 checkpoint。现有 GRU1/GRU3 的 v2 训练继续；LSTM3 v2 会话及其训练/渲染子进程已停止，manifest 为 `cancelled` 且不再出现在 `./progress`。当时观测到 266,240 帧、200,704 帧 checkpoint；后续产物清理见下文。

| 配对 | 模型 | 隐藏宽度×层数 | 完整可训练参数 | 与配对基线之差 |
| --- | --- | ---: | ---: | ---: |
| 单层 | GRU1 | 512×1 | 3,480,775 | 基线 |
| 单层 | GaWF1 | 1160×1 | 3,479,807 | −0.028% |
| 三层 | GRU3 | 331×3 | 3,487,953 | 基线 |
| 三层 | GaWF3 | 653×3 | 3,488,347 | +0.011% |

GaWF1 与 GaWF3 均复用 `scripts/embclip/canonical_core/gawf.py` 中同一个 recurrence，反馈为上一步 detach 的 raw policy logits，episode reset 时清零；GaWF3 直接复用原先通过 SJC 训练 smoke 的 `ThreeLayerGaWFActorCritic`，其源码 hash 与历史 SJC adapter 相同。SJC CPU 检查确认两模型的 reset、门控梯度、参数量与实际 `training_settings.save_interval=100000`。两项独立 2,048 帧 SJC PPO smoke 均正常退出，checkpoint 包含模型、optimizer、scheduler 和 pipeline state：GaWF1 `nav-robothor-embclip-light-gawf1-smoke-sjc-v1-seed0-20261020`（SHA256 `3f6ae0864ce5de2d1261781ee0171b922c8c5fbebbd4ca6ba1d3b0cb7e4bb709`），GaWF3 `nav-robothor-embclip-light-gawf3-smoke-sjc-v1-seed0-20261020`（SHA256 `6db05c67bf8f568fd59d21995a76205d08764f4b17aaf64c46df435d615725a8`）。

独立的 1M 训练 ID 是 `nav-robothor-embclip-light-gawf1-formal1m-sjc-v1-seed0-20261020`（GPU0）和 `nav-robothor-embclip-light-gawf3-formal1m-sjc-v1-seed0-20261020`（GPU1）；21:19 EDT 已启动并在原有进度条登记，21:23 EDT 两项都已有 `TRAIN: 10240 rollout steps` 的真实训练日志。两项在各自新结果 leaf 保留 manifest 哈希约束的配置快照，使用 `scripts/embclip/run_light_formal_sjc_v3.py` 的 checkpoint 恢复机制；1M 是停止点，仍沿用官方 300M 线性学习率计划以便以后按新 ID 延长。SJC 的四项 EmbCLIP 与两项 PointNav 同机叠跑，后续吞吐、100k checkpoint 与 1M 完成需由实际产物确认。部署与取消回执在 `experiments/deployment/sjc-embclip-light4-gawf-20261020.json`；四模型训练 SR/SPL 快照在 `results/navigation-training-metrics-light4-20261020/`，其中 GaWF 两条曲线截至首次 10,240 帧仅各有一个训练日志点。

2026-10-20 21:33 EDT，按用户要求清理已停用 EmbCLIP 训练产物。SJC 上两项旧 200M（GRU3、GaWF3）和三项 LSTM3（轻量 v1、轻量 v2、one-rollout smoke）的完整 `results/<实验ID>/` 目录已删除，包含 checkpoint、`train.log`、TensorBoard events、尝试日志和结果快照；共 157 个文件、212,172,890 字节，删除后五个精确目录均不存在。清理前核实五项无活动进程，活动任务 manifest 无续训引用。实验 manifest 保留历史身份、观测步数及产物已删除标记，不能再从历史 checkpoint 续训。Amarel 的 LSTM3 `62761580` 在排队时取消，耗时 0，只有配置快照、未生成训练日志；空日志目录也已移除。PointNav LSTM2 及当前四项 EmbCLIP 的结果目录未清理。

## 脚本和改动范围

### SJC 四模型 5M 延长与 seed1（2026-10-21）

用户将轻量 EmbCLIP 比较扩展到每模型 5M 全局帧，并为 GRU1、GRU3、GaWF1、GaWF3 各增一项独立 seed1；PointNav 两项保持原 1M 预算。四项 seed0 是各自 1M 训练的续跑，**5M 是累计终点**；四项 seed1 从随机初始化开始。模型、16 个训练环境、128 步 rollout、PPO 配置、300M 步线性学习率计划和约每 100k 帧保存完整状态 checkpoint 均延续原轻量协议。每项有独立结果 leaf、源码哈希及配置快照，可由同一实验 ID 的最新完整 checkpoint 再次续跑。

新实验 ID 统一为 `nav-robothor-embclip-light-<模型>-formal5m-sjc-v1-seed<0|1>-20261021`。单层 GRU/GaWF 放物理 GPU0，三层 GRU/GaWF 放 GPU1。GRU1 和 GRU3 seed0 分别从已核对 SHA256 的 1,001,472 帧完整 checkpoint 启动，00:11 EDT 均已完成一次新 PPO 更新，达到 1,011,712 帧。GaWF1/GaWF3 原 1M 仍运行，两个 seed0 5M 实验以独立队列会话等待各自 1M 会话停止且出现至少 1M 帧完整 checkpoint，届时固定 checkpoint 路径及哈希再启动；等待时不能称为 5M 训练已经运行。四个 seed1 的训练会话均已启动，00:14 EDT 均有 10,240 帧真实 PPO 更新，日志尚无 traceback/OOM。

训练入口为 `scripts/embclip/run_light_5m_sjc.py`，GaWF seed0 等待入口为 `scripts/embclip/wait_light5m_sjc.py`；四个 `scripts/embclip/light5m_*_config.py` 将 AllenAct 真正读取的 `stage.training_settings.save_interval` 设为 100k，并把阶段终点设为 5M。八项已同轮注册 `experiments/monitoring/jobs/`，SJC `./progress` 沿用原进度条显示；详情及启动证据见 `experiments/deployment/sjc-embclip-light5m-20261021.json`。00:09 EDT 六项新训练仍处于模拟器启动阶段；GPU0/GPU1 当时占用 42,923/41,593 MiB（各有 49,140 MiB），整机 load average 273.57，因此吞吐和完成时间均不能据此保证。

2026-10-21 10:06 EDT 更新了四模型、seed0/1 共八条 SJC 训练 SR/SPL 曲线：`results/navigation-training-metrics-light5m-20261021/embclip_training_sr_spl.{png,pdf}`。每条 seed0 曲线按真实全局帧拼接其 1M 来源和 5M 续训 TensorBoard 标量，seed1 单独读取本批 5M 实验；快照 JSON 逐条记录实验 ID、源 event 文件、标量及步数。它们是训练 episode 指标，不是验证集评测或 5M 完成结果。

2026-10-21 21:21 EDT 单独更新四个 seed0 的训练 SR/SPL 曲线：`results/navigation-training-metrics-seed0-20261021/embclip_training_sr_spl.{png,pdf}`；逐点数据、实验 ID 和 event 来源在同目录 `metrics_snapshot.json`。GRU1/GaWF1/GRU3/GaWF3 分别截取到 4,513,792 / 3,469,312 / 4,298,752 / 3,223,552 全局帧，尚未到 5M 终点；各 seed0 曲线接续原 1M 来源。这些数值来自训练 episode，不是验证集 SR/SPL。

2026-10-21 23:18 EDT 的新图在 `results/navigation-training-metrics-mean-std-20261021/embclip_training_sr_spl.{png,pdf}`：四个模型各画一条 seed0/1 的均值，阴影为总体标准差（`ddof=0`）。两个 seed 的训练标量在共同覆盖的全局帧范围内插值到每 50k 帧；不外推，所以各模型曲线止于较慢 seed 已观察到的位置。原始逐 seed 标量、来源及聚合结果保存在同目录 `metrics_snapshot.json`，仍属训练 episode 指标而非验证集分数。用户决定本批四个三层任务（GRU3/GaWF3 seed0/1）结束后不再提交三层 seed2；SJC 轻量版没有 seed2 排队任务，历史 Amarel seed2 已取消记录保持原样。

2026-10-22 用户追加 SJC 轻量版 GRU1/GaWF1 的 seed3、seed4，各从头训练到 5M 全局帧；每张物理卡运行两个模型：GPU0 为两项 seed3，GPU1 为两项 seed4。复用现有 16 环境、128 步 rollout、PPO 与每约 100k 帧完整 checkpoint 协议；新入口 `scripts/embclip/run_light_5m_sjc_seed34.py` 仅接受 seed3/4，独立结果 leaf 和 manifest 已注册到 `./progress`。08:10:45 EDT 四项 tmux 会话存活、进度均为 0/5M，属启动阶段，不代表完成 PPO 更新；详情见 `experiments/deployment/sjc-embclip-light5m-seed34-20261022.json`。

2026-10-22 发现 SJC 的 5M 完成项仍在 `./progress`：AllenAct 最后一次 rollout 合法到达 5,001,216 帧，但旧封装 `verify_checkpoint` 拒绝任何超过 5,000,000 的步数，因此训练子进程结束后没有写 summary/完成 manifest。对已退出的五项（GRU1 seed0/1、GRU3 seed0/1、GaWF1 seed0）逐一核验 checkpoint 的完整模型/优化器/训练管线/调度器状态、seed、步数、SHA256 和训练日志的 clean exit 后，补写完成记录；未伪造缺失的 wrapper 退出码。四项旧 1M seed0 来源已由固定哈希的 5M 续训使用，标为 `superseded`。`scripts/progress.py` 现在即使 5M 后续项已完成，也不会重新显示其旧 1M 来源；`scripts/embclip/reconcile_light5m_sjc.py --watch` 在 SJC 的 `reconcile-embclip5m-20261022` 会话中每 60 秒对剩余 5M 项执行同一完成核验，完成后自动从活动列表移除。08:24 EDT `./progress` 只显示仍在运行的两项 PointNav seed1、三项原 EmbCLIP 5M 和四项新 seed3/4；完整修复与证据见 `experiments/deployment/sjc-embclip-light5m-completion-reconcile-20261022.json`。

2026-10-22 追加单层 GRU1/GaWF1 的独立 seed2，使这两模型各有 seed0–4 共五个 5M 任务。SJC 08:49 EDT GPU0/GPU1 提交前分别剩 14,402/5,714 MiB；GRU1 seed2 在 GPU0 启动，GaWF1 seed2 在 GPU1 以 tmux 队列登记，待 GPU1 空闲显存至少 11,000 MiB 后由 `scripts/embclip/wait_light5m_seed2_gpu1.py` 自动启动。两项复用 16 环境、128 rollout、5M 目标与约每 100k 帧完整 checkpoint；新 `scripts/embclip/run_light_5m_sjc_seed2.py` 明确允许最后一次 rollout 在一个 2,048 帧块内越过目标，5,001,216 帧边界的本机测试通过。完成核验会话已扩展到 seed2。08:49:57 EDT `./progress` 显示 GRU1 seed2 为启动中 0/5M、GaWF1 seed2 排队，不能视作已训练或完成。各自 manifest、配置快照和资源门槛见 `experiments/deployment/sjc-embclip-light5m-seed2-20261022.json`。

### seed0/1 完整验证集评测（2026-10-22）

SJC 已核对四模型各 seed0/1 的八个 5M 训练 manifest 均为完成状态，最终 checkpoint 均为 5,001,216 全局帧。评测使用公开 RoboTHOR ObjectNav **完整 val 集**的 15 个场景、1,800 条唯一 episode，不使用先前的 val100 功能子集。八项评测 ID 为 `nav-robothor-embclip-light-<gru1|gawf1|gru3|gawf3>-eval-val1800-sjc-v1-seed<0|1>-20261022`；各自固定训练 checkpoint 的 SHA256 与 val 数据 15 个文件的 SHA256，结果写入独立 `results/<评测ID>/`。

入口 `scripts/embclip/eval_light5m_sjc.py --gpu 0|1` 使用训练时对应模型配置、官方 AllenAct `--eval`、`test_on_validation=true`、所有模型一致的评测 seed 0、15 个 test process 和官方 500 步 episode 上限。GPU0 顺序运行 GRU1/GaWF1 各 seed0/1，GPU1 顺序运行 GRU3/GaWF3 各 seed0/1；两个队列的 tmux 会话分别为 `embclip-val1800-gpu0-20261022`、`embclip-val1800-gpu1-20261022`。AllenAct 原始指标文件名含 `test`，实际数据集仍是 val。只有 runner 正常退出、1,800 条结果与 val episode ID 集合完全一致，且汇总 SR/SPL 与逐条结果一致时，才将该项 manifest 标为 `complete`。排队和运行状态均不是验证结果。

首批 GRU1 与 GRU3 seed0 的官方 runner 均以返回码 0 结束，但各只产生 1,783 条唯一结果；两项缺少同样的 17 个 val episode ID。原始指标、缺失 ID 和哈希均保留，manifest 标为 `partial`，不能称作完整 1,800 条验证。其余评测继续运行；八项结束后 `scripts/embclip/compare_val1800_sjc.py` 将在全部模型共同实际覆盖的 ID 集合上计算对照分数，并同时报告每项原始覆盖数。官方采样器在初始位姿传送失败时会跳过 episode；本次 17 条缺失的具体原因尚未单独复现。

按用户 2026-10-22 的要求，持续监测会话 `finalize-embclip-val1800-20261022` 已停止；两个评测队列继续运行。需要结果时做单次状态检查；八项均为 `complete` 或 `partial` 后，可单次运行 `python3 -B scripts/embclip/compare_val1800_sjc.py` 生成 `results/embclip-val1800-seed01-20261022/comparison.{json,md}`。任何项失败会停止汇总并保留错误，不能用部分已完成项推出八模型结论。

- `scripts/embclip/install.sh render|full`：两端均已完成full依赖阶段；Amarel随后将AI2-THOR由4.3.0升级至5.0.0，正式训练实际环境已通过`pip check`。
- `scripts/embclip/cache_simulators.py`：仅缓存Unity构建，不启动渲染。
- `scripts/embclip/render_check.py`、`render_job.sh`：计算节点真实RGB/动作自检及限时fallback。
- `scripts/embclip/prepare_assets.py episodes|checkpoint|clip|verify`：SJC已下载并验证独立公开资产；保留原包，检查压缩CRC、类别、scene split和OpenAI RN50官方SHA256。
- `scripts/embclip/sjc_config.py`、`prepare_eval_subset.py`、`eval_sjc.sh`：SJC路径适配、确定性val子集与官方checkpoint评测启动器。评测使用官方runner和配置，子集仅用于功能smoke。
- `scripts/embclip/sjc_train_smoke_config.py`、`train_smoke_sjc.sh`：SJC单阶段训练smoke适配与限时启动器，正式预算和PPO参数不借此改动。
- `scripts/embclip/sjc_resume_smoke_config.py`、`resume_smoke_sjc.sh`：从前述7,680步真实checkpoint继续到下一个rollout的独立恢复验收。
- `scripts/embclip/sjc_profile_smoke_config.py`、`profile_smoke_sjc.sh`、`summarize_profile_sjc.py`：继续到76,800步，在独立结果leaf记录逐卡GPU/整机CPU五秒采样、训练指标与经checkpoint校验的稳定区间吞吐。
- `scripts/embclip/formal_*_config.py`、`canonical_core/`、`formal_amarel_job.sh`及`formal_amarel_node.sh`：三模型正式配置、原样GaWF数学、多节点全局60环境分配和Slurm启动器；`validate_formal_models.py`及SJC新模型smoke提供提交前结构与功能证据。
- `scripts/embclip/sjc_single_gpu_formal.py`、`sjc_{gru3,gawf3}_formal_config.py`及`run_sjc_single_gpu_formal.sh`：仅将正式配置的机器分配改为单张SJC GPU，各模型保持独立60环境、200M步计划，部署到各自结果leaf运行。

新增vendor保持原样，PointNav配置/脚本/历史manifest不修改。仅更新AGENTS中的新支线授权、HM3D取消标记及本支线文件；监控索引在新增实验后重建。根目录无Git元数据，变更清单及SHA256见 `experiments/deployment/amarel-embclip-changes-20261019.json`，可审阅patch见同名 `.diff`（不含整个vendor拷贝；已有审计JSON及生成索引仅列变更范围）。vendor自身git diff为空。


## 2026-10-24 official released checkpoint on the lightweight Val protocol

User authorized evaluating the official public checkpoint on the same Val episodes
as all five GRU1 and five GaWF1 lightweight seeds. No webpage update is authorized.
The released checkpoint has **130,091,717 actual steps**, not a final 200M-step
checkpoint. The paper describes a 200M training budget; the public download is
linked in the official EmbCLIP RoboTHOR README. Its SHA256 is
`b7867c833412212d1921ef3b66baae64d0eec18c6a8825ba4a2c5808407d50a2`.

Run ID: `nav-robothor-embclip-official130m-eval-val1800-sjc-v1-20261024`.
The run uses GPU0, AI2-THOR 5.0.0/build `f0825767cd50d69f666c7f282e54abfe58f1e917`,
15 Val samplers, evaluation seed0, 400×300 raw RGB resized/normalized to224×224,
all 1,800 official Val episodes, and the same 500-step limit and sampled-action
strategy as the ten lightweight evaluations. The official model state loads
strictly. Cache paths are explicit; HOME and the original training sources are
not changed. Frozen runner and config copies are in the independent result leaf.

At 2026-10-24 10:26:52 EDT the run was still running:118/1,800 episodes.
This is a progress snapshot, not a scored result. User chose background continuation
and a later manual result query; no automation or continuing watcher is created.
Launch evidence: `experiments/deployment/sjc-embclip-official130m-val-launch-20261024.json`.

After completion, `scripts/embclip/compare_official_light_val.py` reads all eleven
verified outputs and computes SR/SPL on their common episode IDs. Five-seed
uncertainty is sample std (`ddof=1`); the single official checkpoint has no across-seed
std. Existing ten lightweight outputs were reverified in this turn:1,783 common
Val episodes. The official checkpoint must join that same comparison set before
its score can be placed alongside them. The comparison script only prints JSON.

For clarity, the public 2021 Challenge Test split has10 scenes and2,040 episodes;
scene IDs, goals and initial poses are public. Test object visibility metadata and
shortest paths needed for scoring are intentionally hidden. Train/Val retain those
inputs. EmbCLIP Table1 reports that Challenge Test through the then-operational
organizer scorer; it is not an undisclosed scene/episode split. See
`docs/ROBOTHOR_CHALLENGE2021.md` for the tested scoring limitations.
