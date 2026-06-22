#!/bin/bash  
#SBATCH --job-name=sglopart 
#SBATCH --partition=l40s-gcondo
#SBATCH --time=5:00:00             # total run time limit (DD:HH:MM:SS)  
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)  
#SBATCH --mem=64G        # total memory per node (4 GB per cpu-core is default)  
#SBATCH --gres=gpu:1
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --error=out/sglopart-%A.err ## %A - filled with jobid  
#SBATCH --output=out/sglopart-%A.out ## %A - filled with jobid  
##SBATCH --mail-type=begin       # send email when job begins  
#SBATCH --mail-type=end         # send email when job ends  
#SBATCH --mail-user=shachar_gottlieb@brown.edu

PREFIX=ak8_MD_inclv10_scouting_Upsilon
config=weaver/data_new/UpsilonTo3Gluons/${PREFIX//./_}.yaml
NGPUS=1
label_cls_nodes="['label_Upsilon_ggg','label_QCD']"
load_model="weaver/model/save/2024.pt"

# Change to your conda environment
source /users/sgottli4/miniconda3/etc/profile.d/conda.sh
conda activate weaver

torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS weaver/train.py \
--run-mode "train,val,test" --train-mode hybrid --in-memory \
-o use_swiglu_config True -o use_pair_norm_config True --seed 42 \
-o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[64,64,64]' -o num_heads 16 -o num_layers 10 \
-o reg_kw "{'gamma':2.,'composed_split_reg':[True,False],'as_resid_of':[1]}" \
--use-amp --batch-size 512 --start-lr 8e-6 --num-epochs 20 --optimizer ranger --fetch-by-files --fetch-step 100 --num-workers 4 \
--data-train '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v1/run[1289]/deepntuples/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/QCD/*/run2/deepntuples/job*/*.root' \
--data-test '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/SingleUpsilon/Octet*/run*/deepntuples/job*/*.root' \
	    '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v1/run[1289]/deepntuples/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/QCD/*/run2/deepntuples/job*/*.root' \
-o num_nodes 6 -o num_cls_nodes 2 -o label_cls_nodes ${label_cls_nodes} \
--samples-per-epoch $((1000 * 512)) --samples-per-epoch-val $((200 * 512)) \
--lr-scheduler flat+cos --warmup-steps 2000 \
--data-config ${config} \
--network-config weaver/networks/stage3/example_GloParT3_forScouting.py \
--model-prefix weaver/model/${PREFIX}/ \
--log-file $HOME/scratch/logs/${PREFIX}/train.log \
--tensorboard _${PREFIX} \
--predict-output $HOME/scratch/predict/$PREFIX/pred_CHS.root \
--load-model-weights ${load_model} --exclude-model-weights 'part.fc' \
--freeze-model-weights "(.*blocks\.[0-5]\..*)" 

# Datasets are sitting on BRUX at /HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/run1/deepntuples/job*/*.root, /HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/QCD/QCD*/run1/deepntuples/job*/*.root, /HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/SingleUpsilon/run1/deepntuples/job*/*.root
# Change model prefix to "--model-prefix weaver/model/${PREFIX}/_best_epoch_state.pt" to train starting from previous best epoch
# --freeze-model-weights "(.*embed.*|.*blocks.*)" freezes transformer model weights; can be removed for full training
