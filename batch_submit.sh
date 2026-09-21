#!/bin/bash  
#SBATCH --job-name=sglopart 
#SBATCH --partition=l40s-gcondo
#SBATCH --time=3:00:00             # total run time limit (DD:HH:MM:SS)  
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

set -e 

PREFIX=ak8_MD_inclv10_scouting_Upsilon
config=weaver/data_new/UpsilonTo3Gluons/${PREFIX//./_}.yaml
NGPUS=1
label_cls_nodes="['label_Upsilon_ggg','label_Upsilon_tauhtauh','label_QCD']"
load_model="weaver/model/save/2024.pt"

# Change to your conda environment
source /users/sgottli4/miniconda3/etc/profile.d/conda.sh
conda activate weaver

torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS --max_restarts=0 weaver/train.py \
--run-mode "train,val,test" --train-mode hybrid --in-memory \
-o use_swiglu_config True -o use_pair_norm_config True --seed 42 \
-o fc_params '[(512,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[64,64,64]' -o num_heads 16 -o num_layers 10 \
-o reg_kw "{'gamma':5.,'split_reg':False}" \
--use-amp --batch-size 512 --start-lr 1e-6 --num-epochs 30 --optimizer ranger --fetch-by-files --fetch-step 100 --num-workers 4 \
--data-train '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v[12]/run*/deepntuples/job*/*.root'  \
	    '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/UpsilonToTauHTauH/run[12]/deepntuples/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/QCD/*/run[12]/deepntuples/job*/*.root' \
--data-test '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/SingleUpsilon/upsilon_[123]s/Octet*/run*/deepntuples/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/SingleUpsilonToTauHTauH/upsilon_1s/*/run*/deepntuples/job*/*.root' \
	    '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v[12]/run*/deepntuples/job*/*.root'  \
	    '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/UpsilonToTauHTauH/run[12]/deepntuples/job*/*.root' \
            '/HEP/export/home/jofferma/projects/upsilon3g/UpsilonTo3Gluons/mc/QCD/*/run[12]/deepntuples/job*/*.root' \
-o num_nodes 4 -o num_cls_nodes 3 -o label_cls_nodes "${label_cls_nodes}" \
--samples-per-epoch $((1500 * 512)) --samples-per-epoch-val $((100 * 512)) \
--lr-scheduler flat+decay --optimizer-option weight_decay 1e-4 \
--data-config ${config} \
--network-config weaver/networks/stage3/example_GloParT3_forScouting.py \
--model-prefix weaver/model/${PREFIX}/ \
--log-file $HOME/scratch/logs/${PREFIX}/train.log \
--tensorboard _${PREFIX} \
--load-model-weights ${load_model} --exclude-model-weights 'part.fc' \
--predict-output $HOME/scratch/predict/$PREFIX/pred.root \
--optimizer-option lr_mult '["part\.fc\..*", 20]' \
-o export_params "{'apply_softmax': True, 'num_cls': 2}" \
--freeze-model-weights "(.*embed.*|.*blocks\.[0-9]\..*)"
# --predict
# --export-onnx $HOME/scratch/onnx/2024_scouting.onnx 

# --data-train '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v1/run[1289]/deepntuples_merged.root'  \
#             '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/QCD/QCD_Bin-PT-*_TuneCP5_13p6TeV_pythia8/run2/deepntuples/job*/*.root' \
# --data-test '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/SingleUpsilon/upsilon_*/Octet*/run[1-3]/deepntuples/job*/*.root' \
#             '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/Upsilon_modified_mass/v2/run[12]/deepntuples_merged.root' \
#             '/HEP/export/home/sgottli4/CMSSW_15_0_4/src/UpsilonTo3Gluons/mc/QCD/QCD_Bin-PT-*_TuneCP5_13p6TeV_pythia8/run2/deepntuples/job*/*.root' \

# --lr-scheduler flat+cos --warmup-steps 3000 
# -o reg_kw "{'gamma':5.,'composed_split_reg':[False],'as_resid_of':None}" \
# -o reg_kw "{'gamma':5.,'split_reg':False}" \
# --load-model-weights ${load_model} --exclude-model-weights 'part.fc'
# --freeze-model-weights "(.*blocks\.[0-4]\..*)"
