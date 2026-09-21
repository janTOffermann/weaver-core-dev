#!/bin/bash  
#SBATCH --job-name=sglopart 
#SBATCH --partition=l40s-gcondo
#SBATCH --time=04:00:00             # total run time limit (DD:HH:MM:SS)  
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)  
#SBATCH --mem=32G        # total memory per node (4 GB per cpu-core is default)  
#SBATCH --gres=gpu:1
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --error=out/sglopart-%A.err ## %A - filled with jobid  
#SBATCH --output=out/sglopart-%A.out ## %A - filled with jobid  
##SBATCH --mail-type=begin       # send email when job begins  
#SBATCH --mail-type=end         # send email when job ends  
#SBATCH --mail-user=shachar_gottlieb@brown.edu

PREFIX=ak8_MD_inclv10_scouting_default
config=weaver/data_new/UpsilonTo3Gluons/${PREFIX//./_}.yaml
NGPUS=1
load_model="weaver/model/save/2024.pt"

set -e

# Change to your conda environment
source /users/sgottli4/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Datasets are sitting on BRUX at /HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/run1/deepntuples/job*/*.root, /HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/QCD/QCD*/run1/deepntuples/job*/*.root, /HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/SingleUpsilon/run1/deepntuples/job*/*.root
# Change model prefix to "--model-prefix weaver/model/${PREFIX}/_best_epoch_state.pt" to train starting from previous best epoch
# --freeze-model-weights "(.*embed.*|.*blocks.*)" freezes transformer model weights; can be removed for full training

label_cls_nodes="['label_H_bb','label_H_cc','label_H_ss','label_H_qq','label_H_bc','label_Hp_bc','label_H_bs','label_H_cs','label_Hp_cs','label_Hp_ud','label_Hm_ud','label_H_gg','label_H_ee','label_H_mm','label_H_tauhtaue','label_H_tauhtaum','label_H_tauhtauh','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"

torchrun --standalone --nnodes=1 --nproc_per_node=1 --max_restarts=0 weaver/train.py --run-mode "test" \
-o num_nodes 46 -o num_cls_nodes 22 -o label_cls_nodes ${label_cls_nodes} -o use_swiglu_config True -o use_pair_norm_config True \
-o fc_params '[(2048,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[64,64,64]' -o num_heads 16 -o num_layers 10 \
-o export_params "{'apply_softmax': True, 'num_cls': 22}" \
--num-workers 0 --fetch-by-files --fetch-step 100 \
--network-config weaver/networks/stage3/example_GloParT3_forScouting.py \
--data-test '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/SingleUpsilon/upsilon_[123]s/Octet*/run*/deepntuples_ak4_chs/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/SingleUpsilonToTauHTauH/upsilon_1s/*/run*/deepntuples_ak4_chs/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v[12]/run*/deepntuples_ak4_chs/job*/*.root'  \
            '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/UpsilonToTauHTauH/run[12]/deepntuples_ak4_chs/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/QCD/*/run[12]/deepntuples_ak4_chs/job*/*.root' \
--data-config weaver/data_new/inclv10_aux/ak8_MD_inclv10_scouting_2p.yaml \
--model-prefix weaver/model/save/2024.pt \
--log-file $HOME/scratch/logs/${PREFIX}/train.log \
--tensorboard _${PREFIX} \
--predict-output $HOME/scratch/predict/default_config/pred_pretrain_ak4_chs.root \
--predict
# --export-onnx $HOME/scratch/onnx/2024_scouting.onnx
