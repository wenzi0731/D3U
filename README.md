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

The evaluator now uses the **baseline5 metric definitions**, without changing
D3U training, checkpoint weights, the sampler, or the default sampling RNG.

Use a **new empty output directory** for each evaluation; existing results are
never overwritten. Existing checkpoints remain compatible. New aliases
`--outdir` / `--scenarios`, plus `--batch-size`, `--seed`, and `--no-plots`
are supported. Keep batch size, sampler, steps, scenarios and seed fixed when
reproducing an earlier draw.

Install the added scoring dependencies if necessary (no PyTorch upgrade needed
for this output-only change):

```bash
python -m pip install "scipy>=1.10" "scikit-learn>=1.3"
```

### Output files

- `channel_metrics.csv`: four rows, with MAE, RMSE, MAE_Z, RMSE_Z, R2,
  CRPS, nCRPS, Precision_Z, Recall_Z, CR, IW, IS, IS_Z.
- `global_metrics.json` and `metrics_summary.json`: all metrics and protocol
  metadata, including macro_nCRPS, ES/ES_Z and VS/VS_Z.
- `global_metrics.csv`: **metric,value rows**, matching baseline5.
  The old one-row wide layout is preserved as `global_metrics_wide.csv`;
  update any scripts that previously read the old CSV layout.
- `daily_scores.csv`: date, daily ES_Z/VS_Z and per-channel CRPS, IS, IS_Z.
- `metric_definitions.json`: normalization, score and interval conventions.
- `baseline1_scenarios.npz`: physical scenarios [day, scenario, channel, hour],
  targets, deterministic Stage-1 predictions, dates, channel names, normalized
  conditions, PV-year inputs, **training target_mean/target_std**, seed, sampler,
  scenario count and JSON metadata. Stage-1 predictions are preserved for
  analysis, but the main point metrics use the **scenario mean**.
- `random_timeseries_50/`: up to 50 distinct randomly selected test days.
- `pearson/`: real/generated global Pearson heatmaps and absolute difference.
- `global_pearson.png`: the real-data global correlation heatmap.
  `--no-plots` skips figures and Pearson comparison scores, as in baseline5.

### Metric conventions

The scoring implementation is aligned with CSDI baseline5 commit
`17c217bca33a8e769cbb094c177cfe13c2f298bf`.

- MAE/RMSE/R2 use the ensemble mean, not the Stage-1 deterministic prediction.
  R2 pools all test days/hours per channel; a constant target gives JSON null
  and a blank CSV cell, not a fabricated zero.
- MAE_Z/RMSE_Z use the training-channel z-score; they are NOT min-max scores.
- nCRPS = sum(pointwise empirical CRPS) / (sum(abs(truth)) + 1e-8).
  `macro_nCRPS` is the unweighted mean of four channel scores;
  legacy `mean_nCRPS` is the pooled, scale-weighted score. Do not confuse them.
- CR/IW/IS use a **95%** central interval (linear 2.5% and 97.5% quantiles).
  IS uses alpha=0.05; IS_Z divides the physical IS by the training std.
  Legacy Coverage90/IntervalWidth90 are retained in global outputs separately.
- ES/ES_Z use the same standardized 96-dimensional daily joint vector,
  with the empirical S-squared pairwise denominator.
- VS/VS_Z use p=0.5, channel-major order, **cross-channel pairs only**,
  both same/different hours, weight 1 per unordered pair. Sum all 3456 pairs,
  then average days; do NOT divide by 3456.
- Precision_Z/Recall_Z deliberately retain baseline5's nearest-center-radius
  approximation, not the full union-of-kNN-balls estimator. Defaults: k=5,
  at most 10000 generated daily trajectories per channel, sampled using
  seed + 1000 + channel. Override via optional checkpoint config
  `eval.precision_recall_k` / `eval.max_precision_samples`.
- Generated PV retains the original physical nonnegative clipping.
  Joint scoring standardizes these same postprocessed physical scenarios.
- D3U still resets its global sampling RNG to the checkpoint seed by default.
  It does not adopt baseline5's seed+90000 convention silently. Actual
  sample_seed, training_seed, training_timesteps and sampling_steps are recorded.

### Rescore existing scenarios without resampling

For a new-format archive containing training normalization:

```bash
python -m baseline1.score_npz \
  --npz evaluation_results/baseline1/baseline1_scenarios.npz \
  --outdir evaluation_results/baseline1_rescored
```

For an **old** baseline1 archive lacking target_mean/target_std:

```bash
python -m baseline1.score_npz \
  --npz evaluation_results/baseline1/baseline1_scenarios.npz \
  --checkpoint experiments/baseline1/d3u_condition_only_baseline1_seed42/best_baseline1.pt \
  --outdir evaluation_results/baseline1_rescored
```

Use the original cleaned CSVs. If their locations have changed, supply
`--energy-path` and `--weather-path`. Statistics are refitted only on
2014–2020, and the archived test dates/targets are checked against these CSVs.
Legacy checkpoints do not contain a training-data hash, so this check cannot
prove that their historical training rows are unchanged. Never substitute
statistics fitted on the test set.

Rescoring writes the metric tables, definitions and plots to the new directory;
it leaves the source archive untouched and does not duplicate/resample it.
Its `--seed` controls metric subsampling/plot selection only, not model sampling.

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
