# D3U — HEEW Condition-Only Baseline 1

This fork contains the official D3U implementation and a dedicated adaptation
for comparison with the two-stage joint source–load scenario generator.

> **Method label for papers:** `D3U-Cond (Baseline 1)`
>
> This is an adaptation of D3U to a condition-only scenario-generation task,
> not an exact reproduction of the original history-based forecasting protocol.

The original D3U paper is *Diffusion-based Decoupled Deterministic and Uncertain
Framework for Probabilistic Multivariate Time Series Forecasting* (ICLR 2025).
The original authors and citation are retained below.

## Why an adaptation is required

Original D3U receives historical target sequences. The proposed `2-stages`
method deliberately receives no historical energy. Giving D3U history would
change the information set and invalidate the baseline comparison.

The `baseline1/` entrypoint therefore preserves D3U's defining decomposition:

1. a deterministic condition network predicts the 24-hour four-channel mean;
2. the official PatchDN diffusion model learns the remaining uncertainty;
3. generated residuals are added to the deterministic prediction.

The deterministic network receives only target-day weather, cyclical calendar
features, and the PV-only year control. No historical electricity, heat,
cooling, or PV values are constructed or returned by the dataset.

## Fair-comparison contract

| Item | D3U-Cond Baseline 1 |
|---|---|
| Targets | Electricity, Heat, Cooling, PV |
| Horizon | One complete 24-hour day |
| Train | 2014–2020 |
| Validation | 2021 |
| Test | 2022 |
| Normalization | Fitted on 2014–2020 only |
| Conditions | Same 10 weather + 8 cyclical time features |
| Historical energy | Not used |
| PV year control | PV output head only |
| Default scenarios | 100 |
| Default diffusion setup | 100 steps, `x_start` parameterization |
| Default sampler | 20-step DPM-Solver |
| Seed | 42 |

Observed/reanalysis weather should be described as an oracle-condition setting.
For an operational day-ahead experiment, both methods must receive the same
archived weather forecasts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Place the same cleaned files used by `2-stages` at:

```text
Data/CN03_energy_cleaned.csv
Data/weather_cleaned.csv
```

Required columns are documented in [Data/README.md](Data/README.md).

## Train

Train the deterministic network first, freeze it, then train residual diffusion:

```bash
python -m baseline1.train --config baseline1/configs/heew.yaml --stage all
```

Configuration overrides do not require editing the public YAML:

```bash
python -m baseline1.train \
  --config baseline1/configs/heew.yaml \
  --set train.condition_epochs=2 \
  --set train.diffusion_epochs=2
```

## Generate and evaluate scenarios

```bash
python -m baseline1.evaluate \
  --checkpoint experiments/baseline1/d3u_condition_only_baseline1_seed42/best_baseline1.pt \
  --num-scenarios 100 \
  --sampler dpm_solver \
  --output-dir evaluation_results/baseline1
```

The evaluator saves:

- `baseline1_scenarios.npz` with scenarios `[day, scenario, channel, hour]`,
  targets, deterministic predictions, dates, seed, and channel names;
- `metrics_summary.json` with channel RMSE, MAE, CRPS, nCRPS, 90% coverage,
  interval width, and mean nCRPS.

The NPZ file is intended for the shared final comparison evaluator so all
methods receive identical post-processing and metrics.

## Tests

```bash
pytest
```

Tests cover the fixed year split, train-only normalization, absence of historical
energy inputs, deterministic model shape, and official PatchDN diffusion shape.

## Attribution and citation

Original repository: [Torea-L/D3U](https://github.com/Torea-L/D3U)

```bibtex
@inproceedings{lidiffusion,
  title={Diffusion-based Decoupled Deterministic and Uncertain Framework for Probabilistic Multivariate Time Series Forecasting},
  author={Li, Qi and Zhang, Zhenyu and Yao, Lei and Li, Zhaoxia and Zhong, Tianyi and Zhang, Yong},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

Original contacts:

- Qi Li: li.q@bupt.edu.cn
- Zhenyu Zhang: zhangzhenyucad@bupt.edu.cn

No license file was present in the source repository. Confirm reuse and release
terms before redistributing a modified public release.
