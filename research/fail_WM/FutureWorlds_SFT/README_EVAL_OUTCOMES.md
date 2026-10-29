# RoboCasa outcome evaluation

The existing FutureWorlds evaluator measures causal future-frame quality. The
additional scripts in this directory join those results with the audited
`success` and `failure_type` fields without feeding outcome labels to the model.

## Evaluate one checkpoint

The command exports a checkpoint into an immutable local bundle, evaluates the
held-out validation and test manifests, and writes an outcome report:

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
GPUS=8 ./evaluate_robocasa_checkpoint.sh \
  /media/tyshao/fail_WM/FutureWorlds_SFT/runs/success_only_sft_cached_batched/checkpoints/step_010000 \
  success_step10000
```

For the natural success-plus-failure run:

```bash
GPUS=8 ./evaluate_robocasa_checkpoint.sh \
  /media/tyshao/fail_WM/FutureWorlds_SFT/runs/natural_sft_cached_batched/checkpoints/step_010000 \
  natural_step10000
```

Each run is written under `evals/<name>/`:

```text
bundle/                  exported inference bundle
val/summary.json         validation frame metrics
test/summary.json        test frame metrics
outcome_report.json      machine-readable outcome report
outcome_report.md        comparison-friendly summary
```

The evaluator uses horizons 10, 20, and 32 with full memory and beam width 4.
The outcome report uses LPIPS at horizon 20 by default. The threshold is fitted
on validation only and then applied unchanged to test.

## Compare runs

After both evaluations finish:

```bash
./compare_robocasa_evals.sh success_step10000 natural_step10000
```

The report includes:

- mean success and failure LPIPS;
- error by failure type;
- failure AUROC, where higher visual error is treated as failure risk;
- `optimism_rate`: actual failures classified as success by the validation
  threshold.

Lower failure LPIPS means the predicted rollout is closer to the observed
failed trajectory. A lower `optimism_rate` means fewer failed cases look like
success under this calibrated visual-risk proxy.

This is an outcome-aware prediction evaluation, not a physical robot success
test. A final claim about semantic success prediction still requires a simulator
or state-based goal classifier; no such classifier is silently substituted here.

## RGB outcome judge

`robocasa_video_judge.py` is a provisional task-specific judge for full-length
rollouts. It sees RGB only and detects the red cube and green goal. The success
decision is based on their terminal image-plane distance; the threshold is
calibrated on real validation videos only. With the threshold frozen, the
held-out real-video test set has 99% coverage (99/100), 98/99 correct among
covered cases, and 98/100 correct if unavailable cases count as errors. The
covered-case balanced accuracy is 99.2%. One actual failure has no visible
cube at the end and receives no prediction. Among the 41 actual test failures,
40 are correctly classified and one is unavailable. These real-video numbers
do not establish accuracy on model-generated videos, which can have blur,
color drift, or implausible geometry.

`audit_robocasa_video_judge.py` cross-checks validation/test labels against
the original HDF5 event flags and audits the frozen judge under simple image
degradations. Its report is `evals/real_video_judge_audit.json`. All 200
validation/test labels matched the original HDF5 terminal success flags. On
the same 100 held-out real test videos, accuracy counting unavailable as wrong
fell from 98/100 to 41/100 after 70% brightness, 75/100 after Gaussian blur
(sigma 1.5 pixels), and 56/100 after JPEG quality 50. Most new errors called
real successes failures. These transformations are stress tests, not estimates
of the generated-video distribution. The judge is too sensitive to image
quality to support a model comparison on its own.

After both full-rollout evaluations, independently label the generated test
videos from each model. Blind reviewers should mark visible success, visible
failure, or indeterminate, with a second reviewer resolving disagreements.
Compare the RGB judge against those labels and report coverage, each error
direction, and uncertainty; inspect all unavailable and borderline cases.
The real episode's outcome is not a ground-truth label for its predicted video.

## Stable RoboCasa retraining path

The original RT-1 tokenizer in the initial bundle is not RoboCasa-adapted; its
SHA-256 is identical to the RT-1 tokenizer. Reconstructed RoboCasa frames from
that tokenizer alone have about 18.2 dB terminal PSNR on a ten-case audit, so
world-model SFT should not be judged before codec adaptation.

The staged experiment now prepared in this directory is:

1. Adapt the visual tokenizer on the 443 successful training episodes with the
   paper's RoboCasa codec recipe: seven targets spanning gaps 1--40, global
   batch 32, learning rate `5e-5`, 5,000 warmup steps, L1 + VGG-LPIPS, and the
   discriminator after step 10,000 through step 50,000. The 8-card launcher
   uses gradient accumulation to preserve global batch 32.
2. Freeze that adapted tokenizer and run 50,000-step teacher-forced SFT on
   `train_success.json`, global batch 64, backbone LR `1e-5`, text-adapter LR
   `1e-4`, cosine decay, 500 warmup steps, and checkpoints every 5,000 steps.
3. Audit codec reconstruction and short/medium/long SFT validation metrics.
   Only after this baseline is visually stable should failure episodes be added
   in a separate fine-tuning run.

The prepared launchers are `run_paper_robocasa_codec_8gpu.sh` and
`run_paper_robocasa_sft_8gpu.sh`. The first must finish before the second; the
adapted codec is installed into a new verified bundle between them.

For the controlled tokenizer-swap pilot, use `run_paper_robocasa_codec_8gpu.sh`
first, then `run_tokenizer_swap_sft_8gpu.sh`. The latter selects the best
validated codec checkpoint, requires at least 1 dB PSNR improvement and a
terminal horizontal-gradient ratio of 0.8, installs it in an isolated verified
bundle, rebuilds the success-only token cache, and repeats the original
cached-batched SFT configuration at 10,000 steps, global batch 64, micro-batch
8 per GPU. It refuses to start SFT if the codec gate fails. The new SFT run is
`runs/success_only_sft_new_codec_10k`; the old RT-1 codec and cache are not
overwritten. The 50,000-step SFT launcher above is a separate later experiment.
On an 8-GPU host that accepts only one submitted command, run
`./run_tokenizer_swap_experiment_8gpu.sh`. It executes both stages in order and
stops before SFT if the adapted tokenizer fails its reconstruction gate.

The fixed-horizon evaluator is useful for frame-quality curves, but its 32-frame
videos usually stop before the task reaches its terminal state. For semantic
outcomes, use the full-rollout wrapper:

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
GPUS=8 ./evaluate_robocasa_full_checkpoint.sh \
  /media/tyshao/fail_WM/FutureWorlds_SFT/runs/success_only_sft_cached_batched/checkpoints/step_010000 \
  success_step10000_full
```

To infer both trained checkpoints in one 8-GPU job:

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
GPUS=8 ./infer_two_robocasa_checkpoints.sh
```

The script predicts each validation and test episode through its audited final
frame. Outputs are under `evals_full/success_step10000_full/` and
`evals_full/natural_step10000_full/`. Each split has `summary.json`, and each
case has `prediction.npz` and `result.json`. Finished cases are skipped on
restart after the bundle and evaluation protocol are verified. RGB judging is
disabled by default; `RUN_JUDGE=1` enables the provisional judge after all
predictions finish. A predicted video's success must be assessed separately.

## VLM cross-check

The local Qwen2.5-VL 3B checkpoint is installed at:

```text
/media/tyshao/fail_WM/FutureWorlds_workspace/assets/qwen2.5-vl-3b-instruct
```

It reviews only borderline or unavailable RGB cases selected by the geometric
judge. It is not used to fit the threshold or establish accuracy on its own:

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
MAX_CASES=20 ./vlm_review_full_run.sh success_step10000_full
MAX_CASES=20 ./vlm_review_full_run.sh natural_step10000_full
```

The output is `evals_full/<name>/vlm_review.json`. Each answer contains
`success`, `cube_on_green_target`, `gripper_released`, `stable_at_end`, a
failure type, and the raw VLM response. Review disagreements between this file
and `video_judge.json` manually before making a semantic claim.
