# 当前代码入口与文件树

2026-10-27：`scripts/` 按 training / analysis 分开，源码由原来的 **153 个 → 首轮 42 个 → 重组后 35 个 → 当前 30 个**。其中 training 14 个、analysis 16 个，全部为主工作流实现文件。计数仅包括 Python 源文件，不含 macOS 目录元数据。

## 职责与合并

- `training/embclip`：统一五个 seed（代码索引 0–4）的训练、模型配置、模型 adapter 和资产准备，共 4 个文件。
- `training/pointnav`：RGB-only 配置、兼容层、模型、完整状态 trainer、训练 loop 和启动器，共 6 个文件。共享 recurrent 数学在 `training/canonical_core`，不复制到任务 wrapper。
- `analysis/embclip`：评测、统计、expert reference、Grad-CAM、媒体和绘图各一个入口，共 6 个文件。统计共用 manifest 选择与带 hash/身份检查的结果加载器；原候选/十二类分析、选择合并到 `gradcam.py`；录制/重放/启动合并到 `media.py`；validation/training/Grad-CAM/teaser 合并到 `plots.py`。
- `analysis/pointnav`：评测、统计和绘图各一个入口。项目页构建调用两任务的统一绘图函数，不再保留第二份 plotting 实现。
- training 和 analysis 各有自己的监控入口。训练代码不导入 analysis；analysis 可使用训练模型和调度查询工具。

## 完整文件树

```text
scripts/
├── analysis/
│   ├── embclip/
│   │   ├── evaluate.py
│   │   ├── expert.py
│   │   ├── gradcam.py
│   │   ├── media.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── pointnav/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── project/
│   │   ├── build_assets.py
│   │   ├── build_page.py
│   │   ├── build_table.py
│   │   ├── method_figure.py
│   │   └── preview.py
│   ├── monitor.py
│   └── plotting.py
└── training/
    ├── canonical_core/
    │   ├── __init__.py
    │   ├── gawf.py
    │   └── rnn.py
    ├── embclip/
    │   ├── config.py
    │   ├── models.py
    │   ├── prepare_assets.py
    │   └── train.py
    ├── pointnav/
    │   ├── compat.py
    │   ├── models.py
    │   ├── run.py
    │   ├── runtime.py
    │   ├── train.py
    │   └── trainer.py
    └── monitor.py
```

## 命令

以下命令从仓库根目录执行，在对应已准备的运行环境中运行仿真任务：

```sh
python -B -m scripts.training.embclip.train <training-id>
python -B -m scripts.training.pointnav.run <training-id>
python -B -m scripts.analysis.embclip.evaluate <evaluation-id> --gpu 0 --validate-only
python -B -m scripts.analysis.pointnav.evaluate <evaluation-id>
python -B -m scripts.analysis.embclip.metrics validation
python -B -m scripts.analysis.embclip.metrics training
python -B -m scripts.analysis.embclip.metrics compare
python -B -m scripts.analysis.embclip.metrics behaviour --reference <expert-result-leaf>
python -B -m scripts.analysis.pointnav.metrics
python -B -m scripts.analysis.embclip.plots validation --source <snapshot.json> --output <output-stem>
python -B -m scripts.analysis.embclip.plots gradcam --source <result-leaf> --output <figure.png> --page 1
python -B -m scripts.analysis.pointnav.plots training --source <snapshot.json> --output <output-stem>
python -B -m scripts.analysis.project.build_assets
python -B -m scripts.analysis.project.preview
./progress --host sjc-remote
./progress analysis --host sjc-remote
python -B -m scripts.training.monitor --rebuild
```

EmbCLIP 的两个模型共用 `training/embclip/config.py`，由 runner 传入 `model=gru1|gawf1`。官方 130M GRU checkpoint 共用 analysis 的评测入口，由 manifest 的 `source_checkpoint.actual_training_steps=130091717` 选择 reference 校验。五个训练 seed 和两模型始终通过同一个 manifest selector 读取，不按日期分批实现；训练 seed0 的历史 1M continuation 身份保留，其余 seed 独立初始化。

媒体操作为 `media record|replay|launch`，Grad-CAM 操作为 `gradcam gradcam-replay|gradcam-sample|select`。`launch` 启动冻结 worker 时，新的 manifest 必须在 `execution.operation` 中明确指定 `record`、`replay`、`gradcam-replay` 或 `gradcam-sample`；worker 与其配置按当前路径/hash 冻结。重放直接复用媒体模块内的录制 helper，不再复制 `recording_helpers.py`。旧结果中的冻结 helper 仍是历史工件，未作改写。

## 引用与验证

历史 manifest、source hash、实验 ID、result/config snapshot、checkpoint 以及日期记录中的原始路径保持原样；旧协议 `.sha256` 文件与 `project-page/dist/assets/code/` 均为历史身份记录，不作为当前入口。新任务必须登记当前路径和 source hash，不能通过改写历史 hash 使旧任务冒充新实现。

```sh
python -B -m pyflakes scripts
```

开发回归测试目录已移除，训练、评测、统计与绘图入口不依赖这些测试文件。本次删除通过主工作流源码 hash 对比、Python 语法与静态检查、CLI 入口加载和引用核对进行验证；30 个主工作流源文件内容保持不变。

此次提交不修改实验数据、checkpoint、结果、环境或 vendor；不部署到 SJC 执行主机、不重新训练、不发布网页。代码提交推送至 Git 的 `sjc` 分支。未在本轮重跑 GPU 仿真，离线测试不等价于远端 smoke。
