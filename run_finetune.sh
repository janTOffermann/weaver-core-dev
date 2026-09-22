#!/bin/bash
#SBATCH --job-name=sglopart-ft
#SBATCH --partition=l40s-gcondo
#SBATCH --time=8:00:00             # bump if you run both stages in one job (STAGE=all)
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --error=out/sglopart-ft-%A.err
#SBATCH --output=out/sglopart-ft-%A.out
##SBATCH --mail-type=end
##SBATCH --mail-user=<your-address>   # uncomment + fill in if you want completion mail
#
# Two-stage fine-tuning of ScoutingGloParT for Y->ggg vs QCD (2 classes)
# + one mass-regression node (target_res_mass_factor = gen_mass / scoutfj_mass).
#
#   Stage 1  head warm-up : backbone fully frozen, only part.fc trainable, high LR, few epochs.
#   Stage 2  discriminative fine-tune : everything trainable, transformer backbone at 10%
#            of the head LR via the optimizer `lr_mult` option.
#
# Usage:   sbatch run_finetune.sh 1      # stage 1 only
#          sbatch run_finetune.sh 2      # stage 2 only (needs stage 1's best checkpoint)
#          sbatch run_finetune.sh all    # both, sequentially  (default)
#          bash   run_finetune.sh 1      # interactively, if you already hold a GPU
#
# NOTE: `mkdir -p out` before the first sbatch -- SLURM opens the log files before
#       this script runs, so it cannot create that directory for you.

set -e

STAGE="${1:-all}"

# =============================== configuration ===============================

PREFIX=ak8_MD_inclv10_scouting_Upsilon
CONFIG=weaver/data_new/UpsilonTo3Gluons/${PREFIX//./_}.yaml
NETWORK=weaver/networks/stage3/example_GloParT3_forScouting.py

# GloParT v3 scouting pre-training: 252 tensors, part.fc.1 is (46, 2048).
PRETRAINED=weaver/model/save/2024.pt

NGPUS=1
CONDA_ENV=weaver
CONDA_SH=/users/jofferma/opt/miniconda3/etc/profile.d/conda.sh

# --- input data ---------------------------------------------------------------
# Do NOT merge the per-mass-point signal files. Each fetch reads the same entry window
# from EVERY file in the worker's list (dataset.py:198-209), so one file per mass point
# makes every fetch automatically stratified across all mass points. Merging replaces
# that with a contiguous window of the concatenation, i.e. a few mass points per fetch.
# Keep the file count >= --num-workers: actual_workers = min(num_workers, n_files)
# (train.py:276), and dataset.py:153 asserts each group gets at least one file.
JET_COLLECTION="deepntuples"


DATA_DIR=/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc

TRAIN_SIG="${DATA_DIR}/Upsilon_modified_mass/v[12]/run[1289]/${JET_COLLECTION}/job*/*.root"
TRAIN_BKG="${DATA_DIR}/QCD/QCD_Bin-PT-*_TuneCP5_13p6TeV_pythia8/run2/${JET_COLLECTION}/job*/*.root"

TEST_SIG_OCTET="${DATA_DIR}/SingleUpsilon/upsilon_*/Octet*/run*/${JET_COLLECTION}/job*/*.root"
TEST_SIG_SCAN="${DATA_DIR}/Upsilon_modified_mass/v2/run3/${JET_COLLECTION}/job*/*.root"
TEST_BKG="${DATA_DIR}/QCD/QCD_Bin-PT-*_TuneCP5_13p6TeV_pythia8/run2/${JET_COLLECTION}/job*/*.root"

# --- node counts --------------------------------------------------------------
# num_cls_nodes = 2 (Y->ggg, QCD); num_nodes = 2 cls + 1 regression = 3.
NUM_CLS_NODES=2
NUM_NODES=3
LABEL_CLS_NODES="['label_Upsilon_ggg','label_QCD']"

# gamma balances CrossEntropy against gamma * LogCosh (see HybridLoss in the network
# config). 5 was tuned for the 46-node/24-regression-target pre-training; with a single
# regression target you will likely want it lower. Watch Loss/train vs LossReg/train.
REG_KW="{'gamma':5.,'split_reg':False}"

# --- schedule -----------------------------------------------------------------
BATCH=512

S1_EPOCHS=4
S1_SAMPLES=$((500 * BATCH))
S1_LR=2e-3                      # head only; it is randomly initialised, so this is high

S2_EPOCHS=30
S2_SAMPLES=$((1500 * BATCH))
S2_LR=1e-4                      # <-- BACKBONE LR. The head gets S2_LR * S2_HEAD_MULT.
S2_HEAD_MULT=10                 #     10 => head 1e-3, backbone 1e-4 (backbone at 10%)
S2_WARMUP=2000                  # steps; steps_per_epoch = S2_SAMPLES / BATCH = 1500

VAL_SAMPLES=$((100 * BATCH))

MODEL_DIR=weaver/model
LOG_DIR=$HOME/scratch/logs
PRED_DIR=$HOME/scratch/predict
MASSREG_DIR=$HOME/scratch/massreg

# --- regressed-mass histogram metrics (weaver/utils/nn/mass_hist.py) -----------
#
# Each block is a (sample, truth class) selection split into histograms along one
# axis. `sample` is the per-file tag injected by the filepath rules in
# weaver/utils/data/fileio.py -- nominal = SingleUpsilon*, modified =
# Upsilon_modified_mass -- and requires `sample_kind` in the yaml's observers.
# `cls: 0` is Y->ggg (fj_label_scout), which holds for both the 2- and 3-class
# setups since label_cls_nodes only names outputs, it does not remap truth labels.
# `None` as the last pT edge means open-ended; the config goes through
# ast.literal_eval, which has no spelling for infinity.
#
# A block whose sample is absent from the data being evaluated simply does not
# fill, so the same spec serves the in-training pass and mass_reg_posthoc.py.
# Validation is split out of --data-train (smeared-mass only), so in training only
# `smeared_by_mass` fills; `nominal_by_pt` comes from the post-hoc pass over
# --data-test, where the SingleUpsilon samples live.
#
# write_plots is False because this env has no PyROOT: epochs accumulate into
# massreg_hists.npz, rendered afterwards from the `weaver-root` env. See
# "after training" at the bottom of this file.
mass_hist_kw() {  # $1 = stage tag, so each stage keeps its own epoch axis
  cat <<EOF
{'outdir': '${MASSREG_DIR}/${PREFIX}_$1', 'tag': 'massreg', 'write_plots': False,
 'mass_bins': (100, 0., 50.), 'resp_bins': (120, 0., 3.),
 'baseline_epoch': -1, 'min_entries': 50,
 # Which metric decides the "best" epoch. Uncomment to take best-epoch selection
 # away from the validation loss and give it to a metric scoped to ONE block, and
 # therefore ONE sample -- so adding other samples to --data-val cannot change it.
 # Naming two blocks is deliberately not expressible: combining a nominal-mass and
 # a smeared-mass number into one score is not meaningful.
 #   metric: 'abs_median_bias' (|median(pred/true) - 1|), 'resolution', or
 #           'ks_vs_baseline'; lower is better for all three, as train.py expects.
 #   reduce: 'mean' or 'worst' across the block's sub-bins; add 'sub': '<name>'
 #           to pin it to one sub-bin instead.
 # 'selection_metric': {'block': 'smeared_by_mass', 'metric': 'abs_median_bias', 'reduce': 'mean'},
 'blocks': [
   {'name': 'nominal_by_pt', 'sample': 'nominal', 'cls': 0,
    'title': 'Nominal-mass Y#rightarrowggg',
    'split': {'var': 'gen_pt', 'edges': [50, 100, 200, None], 'label': 'Y p_{T}'}},
   {'name': 'smeared_by_mass', 'sample': 'modified', 'cls': 0,
    'title': 'Smeared-mass Y#rightarrowggg',
    'split': {'var': 'gen_mass', 'points': [6, 15, 24], 'tol': 0.5, 'label': 'Y mass'}}]}
EOF
}

# ============================== regex reference ===============================
#
# Verified against the instantiated model (231 parameter tensors):
#
#   part\.fc\..*            ->   4 tensors,  0.53M params: the MLP head
#                                (part.fc.0.0 = 256->2048, part.fc.1 = 2048->3)
#   (?!part\.fc\.).*        -> 227 tensors, 13.71M params: input_embeds.*,
#                                part.pair_embed.*, part.blocks.0-9.*,
#                                part.cls_blocks.0-1.*, part.norm.*, part.cls_token
#   part\.blocks\..*        -> 150 tensors, 10.53M params: the 10 encoder blocks only
#   part\.blocks\.[0-4]\..* ->  75 tensors,  5.27M params: bottom 5 encoder blocks
#                                (for gradual unfreezing)
#   (model total: 231 tensors, 14.24M params)
#
# Careful: a bare `.*blocks\.[0-5]\..*` ALSO matches part.cls_blocks.0/1 -- anchor it.
#
# `--freeze-model-weights` is applied in model_setup() BEFORE the optimizer is built
# (train.py:759), and optim() skips params with requires_grad=False (train.py:451), so
# freezing composes correctly with lr_mult.
#
# Do NOT use `--optimizer-option freeze`: it sits in an elif (train.py:491) and is
# silently ignored whenever weight_decay or lr_mult is set.
#
# The double backslashes in the lr_mult value below are deliberate. The option value is
# parsed with ast.literal_eval (train.py:441); a single backslash gives a
# SyntaxWarning today and will be a hard SyntaxError on Python 3.14.
#
# If you ever add `--backend nccl` for multi-GPU: optim() runs AFTER the DDP wrap
# (train.py:992-995), so every parameter name gains a `module.` prefix and the lr_mult
# pattern must become "module\\.part\\.fc\\..*". It fails silently otherwise -- check the
# log for the "Parameters with lr multiplied by" line. --freeze-model-weights is
# unaffected (it runs pre-wrap).
#
# =============================================================================

source ${CONDA_SH}
conda activate ${CONDA_ENV}

# Arguments shared by both stages: architecture must match the pre-trained checkpoint.
#
# NOTE: --in-memory is deliberately NOT used. With it, only the FIRST fetch is ever
# consumed -- dataset.py:245-250 short-circuits to reshuffling the cached index array
# and the prefetched second fetch (dataset.py:263) is discarded forever. Combined with
# --fetch-step 0.05 that would silently restrict training to the first 5% of entries in
# every file (the offset is deterministic: --data-fraction 1 makes the random offset
# np.random.uniform(0, 0) == 0).
#
# Without --in-memory, fetch_step is just a streaming buffer size: the 16 slices tile
# the whole (0, train_val_split) range, and infinity_mode -- on, because
# --samples-per-epoch is set (train.py:264) -- restarts the cycle when they are
# exhausted (dataset.py:279-282). So every training event is used over the run.
#
# The fetched range is a CONTIGUOUS entry window per file (fileio.py:276-281);
# shuffling and the reweighting resample happen afterwards, within the fetched block
# only (dataset.py:102-110).
COMMON=(
  --train-mode hybrid --seed 42
  --use-amp --batch-size ${BATCH} --optimizer ranger
  --fetch-step 0.05 --num-workers 4
  -o use_swiglu_config True -o use_pair_norm_config True
  -o fc_params '[(2048,0.1)]' -o embed_dims '[256,1024,256]'
  -o pair_embed_dims '[64,64,64]' -o num_heads 16 -o num_layers 10
  -o num_nodes ${NUM_NODES} -o num_cls_nodes ${NUM_CLS_NODES}
  -o label_cls_nodes "${LABEL_CLS_NODES}"
  -o reg_kw "${REG_KW}"
  -o export_params "{'apply_softmax': True, 'num_cls': ${NUM_CLS_NODES}, 'compress_outputs': False}"
  --data-config ${CONFIG}
  --network-config ${NETWORK}
  --samples-per-epoch-val ${VAL_SAMPLES}
)

DATA_TRAIN=(--data-train "${TRAIN_SIG}" "${TRAIN_BKG}")
DATA_TEST=(--data-test "${TEST_SIG_OCTET}" "${TEST_SIG_SCAN}" "${TEST_BKG}")

# ============================== STAGE 1: head warm-up ========================
#
# Load everything from the pre-trained model except the final 46-node output layer:
# `part\.fc\.1` drops exactly 2 of the 252 checkpoint tensors, so the pre-trained
# 256->2048 layer (part.fc.0.0) is kept as the head's initialisation and only the
# 2048->3 output is random. Precedent: train.py:725-732.
#
# The whole backbone is frozen, so only ~0.53M of 14.2M parameters train here.
#
# CHECK THE LOG: "Model initialized with weights from ... Missing: [...]" must list
# ONLY part.fc.1.weight and part.fc.1.bias, and Unexpected must be empty. Anything
# else means the -o architecture flags no longer match the checkpoint.
#
# Note: freezing stops gradients but NOT BatchNorm running-stat updates -- the input_bn
# layers in input_embeds.* and pair_embed will still adapt to your sample during this
# stage. That is usually what you want for the scouting-jet domain shift.

run_stage1() {
  echo "=== STAGE 1: head warm-up (backbone frozen) ==="
  torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} --max_restarts=0 weaver/train.py \
    --run-mode "train,val" \
    "${COMMON[@]}" "${DATA_TRAIN[@]}" \
    --num-epochs ${S1_EPOCHS} --samples-per-epoch ${S1_SAMPLES} \
    --start-lr ${S1_LR} --lr-scheduler flat+decay \
    --optimizer-option weight_decay 1e-4 \
    --load-model-weights ${PRETRAINED} --exclude-model-weights 'part\.fc\.1' \
    --freeze-model-weights "(?!part\.fc\.).*" \
    -o eval_kw "{'mass_hist_kw': $(mass_hist_kw stage1)}" \
    --model-prefix ${MODEL_DIR}/${PREFIX}_stage1/net \
    --log-file ${LOG_DIR}/${PREFIX}_stage1/train.log \
    --tensorboard _${PREFIX}_stage1
}

# ============================ STAGE 2: discriminative LR =====================
#
# Resume from stage 1's best epoch -- the architecture is now identical, so the whole
# state_dict loads and there is no --exclude-model-weights. Nothing is frozen.
#
# lr_mult takes ONE [regex, factor] pair and gives the matched parameters
# lr = start_lr * factor. Putting the HEAD in the multiplied group and leaving
# --start-lr as the backbone LR gives the 10:1 ratio you want:
#
#     backbone (227 tensors) = S2_LR                = 1e-4
#     head     (4 tensors)   = S2_LR * S2_HEAD_MULT = 1e-3
#
# flat+cos is used deliberately instead of the default flat+decay. With lr_mult set,
# flat+decay builds a LambdaLR with lambdas (1, 1, get_lr, get_lr) at train.py:543-547,
# so ONLY the multiplied groups anneal and the 1x groups sit at their initial LR for the
# entire run. flat+cos applies a single lr_fn to every group (train.py:552-575), so both
# anneal together and the 10:1 ratio is preserved throughout.
#
# Also avoid --lr-scheduler one-cycle here: OneCycleLR takes the scalar max_lr=start_lr
# and flattens every group to the same LR, silently discarding the ratio.

run_stage2() {
  echo "=== STAGE 2: full fine-tune, backbone LR = ${S2_LR}, head LR = ${S2_LR} x ${S2_HEAD_MULT} ==="
  torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} --max_restarts=0 weaver/train.py \
    --run-mode "train,val,test" \
    "${COMMON[@]}" "${DATA_TRAIN[@]}" "${DATA_TEST[@]}" \
    --num-epochs ${S2_EPOCHS} --samples-per-epoch ${S2_SAMPLES} \
    --start-lr ${S2_LR} --lr-scheduler flat+cos --warmup-steps ${S2_WARMUP} \
    --optimizer-option weight_decay 1e-4 \
    --optimizer-option lr_mult '["part\\.fc\\..*", '"${S2_HEAD_MULT}"']' \
    --load-model-weights ${MODEL_DIR}/${PREFIX}_stage1/net_best_epoch_state.pt \
    -o eval_kw "{'mass_hist_kw': $(mass_hist_kw stage2)}" \
    --model-prefix ${MODEL_DIR}/${PREFIX}_stage2/net \
    --log-file ${LOG_DIR}/${PREFIX}_stage2/train.log \
    --tensorboard _${PREFIX}_stage2 \
    --predict-output ${PRED_DIR}/${PREFIX}_stage2/pred.root
}

case "${STAGE}" in
  1)   run_stage1 ;;
  2)   run_stage2 ;;
  all) run_stage1; run_stage2 ;;
  *)   echo "Usage: $0 [1|2|all]" >&2; exit 1 ;;
esac

echo "=== done (stage: ${STAGE}) ==="

# ============================== variants =====================================
#
# Freeze the transformer blocks entirely for the whole of stage 2 (instead of the 10%
# LR) -- add to run_stage2 and drop the --optimizer-option lr_mult line:
#
#     --freeze-model-weights "part\.blocks\..*"
#
# Freeze only the bottom half of the encoder (gradual unfreezing):
#
#     --freeze-model-weights "part\.blocks\.[0-4]\..*"
#
# Freeze the whole backbone (identical to stage 1):
#
#     --freeze-model-weights "(?!part\.fc\.).*"
#
# Multiple patterns are comma-separated, so avoid {m,n} quantifiers inside them.
#
# ============================== after training ===============================
#
# Predictions land in ${PRED_DIR}/${PREFIX}_stage2/pred.root -> Run_metrics.ipynb.
# The regression node is a RATIO: reconstruct the mass as
#     output_target_res_mass_factor * scoutfj_mass
#
# --- regressed-mass histograms -------------------------------------------------
#
# Training fills only `smeared_by_mass` (validation is split out of --data-train).
# To get the nominal-mass histograms and the pre-finetune reference, run the
# post-hoc pass. Submit these via sbatch -- they run model inference, so not on
# the login node. Steps 1-2 need the `weaver` env, step 3 needs `weaver-root`,
# and all three must be given the SAME -o eval_kw.
#
# Use a mass-histogram-specific data spec rather than ${DATA_TEST[@]}: the
# smeared-mass runs differ in how far their smearing extends, and the two runs
# held out of TRAIN_SIG both stop at ~20 GeV --
#
#     v1/run1,2,8,9  max gen_mass ~25   v2/run3  max ~20.1
#     v2/run1,2      max gen_mass ~25   v2/run4  max ~19.9
#     v3/run1,2      max gen_mass ~22.6
#
# so TEST_SIG_SCAN (v2/run3) can never fill the 24 GeV sub-bin. The runs that do
# reach 24 GeV are in TRAIN_SIG, but that is fine here: the data config holds out
# by EVENT, not by file -- training takes event_no%10 <= 7 and test_time_selection
# takes event_no%10 > 7 -- so pointing this pass at them still evaluates on events
# the network never trained on. (In-training validation is an entry-range split of
# TRAIN_SIG, which does reach 25 GeV, so the 24 GeV bin fills there already.)
#
#   MASSREG_TEST=(--data-test \
#     "${DATA_DIR}/SingleUpsilon/upsilon_[123]s/Octet*/run*/${JET_COLLECTION}/job*/*.root" \
#     "${DATA_DIR}/Upsilon_modified_mass/v1/run[1289]/${JET_COLLECTION}/job*/*.root" \
#     "${DATA_DIR}/Upsilon_modified_mass/v2/run[12]/${JET_COLLECTION}/job*/*.root")
#
#   EK="{'mass_hist_kw': $(mass_hist_kw stage2)}"
#
#   # 1. pre-finetune reference -> epoch -1, the KS baseline. Note it loads the
#   #    stage-1 checkpoint the way stage 2 starts from it, so "baseline" means
#   #    "where stage 2 began", not the untouched 2024.pt.
#   python weaver/scripts/mass_reg_posthoc.py "${COMMON[@]}" "${MASSREG_TEST[@]}" \
#     -o eval_kw "${EK}" --baseline \
#     --load-model-weights ${MODEL_DIR}/${PREFIX}_stage1/net_best_epoch_state.pt \
#     --model-prefix ${MODEL_DIR}/${PREFIX}_stage2/net
#
#   # 2. every epoch checkpoint (omit --epochs to pick up all that exist)
#   python weaver/scripts/mass_reg_posthoc.py "${COMMON[@]}" "${MASSREG_TEST[@]}" \
#     -o eval_kw "${EK}" --epochs 0-$((S2_EPOCHS - 1)) \
#     --model-prefix ${MODEL_DIR}/${PREFIX}_stage2/net
#
#   # 3. render to .root + .pdf (needs PyROOT, not torch, not the data)
#   conda activate weaver-root
#   python weaver/scripts/mass_reg_posthoc.py -o eval_kw "${EK}" --from-state
#
# Output lands in ${MASSREG_DIR}/${PREFIX}_stage2/: massreg.root (every TH1/TH2/
# TGraph), one multi-page massreg_<block>.pdf per block, and massreg_hists.npz
# (the accumulated counts -- re-runnable, and what step 3 reads).
#
# ONNX export (see run.sh for the full form). compress_outputs MUST be False: the
# compression path in ParticleTransformer2024Plus_forScouting.py hard-codes the 22-class
# GloParT index map and will not work with 2 classes.
#
#   python weaver/train.py --gpus '' --run-mode export_onnx \
#     --export-onnx weaver/model/save/${PREFIX}_finetuned.onnx \
#     -o export_params "{'apply_softmax': True, 'num_cls': 2, 'compress_outputs': False}" \
#     ... same -o architecture flags as above ... \
#     --data-config ${CONFIG} --network-config ${NETWORK} \
#     --model-prefix ${MODEL_DIR}/${PREFIX}_stage2/net_best_epoch_state.pt
