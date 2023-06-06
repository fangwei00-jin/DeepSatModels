# run this file
# bash ./run_in_parallel.sh >./runlog/nohup.log 2>&1 &

# check shell
# ps aux|grep $USER|grep "python -u"

# run python in shell, test it
# python -u ./sleeptest.py > ./runlog/sleep30.out 2>&1
# python -u ./sleeptest.py > ./runlog/sleep31.out 2>&1
{
    # [MFT_base, MFT_FAD_base, MFT_CAT_T, MFT_FAD_T, MFT_FAD_TF_test, MFT_CAT_ViT, MFT_ViT])]
    # ["11x11_Trento", "11x11_Houston", "11x11_MUUFL", "JointTrento", "JointAugsburg"]
    time=$(date "+%m%d%H%M") # get current time # xxx blank matters a lot in bash.

    # command="python -u mft_main.py -m 2 --cuda 3 --LiDAROnly True -n TrentoLOnly"
    # echo $command > ./runlog/${time}TrentoLOnly.out
    # $command >> ./runlog/${time}TrentoLOnly.out 2>&1
}


time=$(date "+%m%d%H%M") # get current time # xxx blank matters a lot in bash.
gpu_ids="1"
dataset="PASTIS24"

command="python -u train_and_eval/segmentation_training_transf.py --config_file configs/PASTIS24/TSViT_fold1.yaml --gpu_ids ${gpu_ids}"
log="./runlog/${time}_data_${dataset}_1.out"
echo $command > $log
echo "run TSViT_fold3 with tsvit" > $log # 2023-5-15 07:43:00
nohup $command >> $log 2>&1 &


time=$(date "+%m%d%H%M") # get current time # xxx blank matters a lot in bash.
gpu_ids="2"
dataset="PASTIS24"

command="python -u train_and_eval/segmentation_training_transf.py --config_file configs/PASTIS24/TSViT_fold3.yaml --gpu_ids ${gpu_ids}"
log="./runlog/${time}_data_${dataset}_3.out"
echo $command > $log
echo "run TSViT_fold3" > $log # 2023-5-15 07:37:00
nohup $command >> $log 2>&1 &


time=$(date "+%m%d%H%M") # get current time # xxx blank matters a lot in bash.
gpu_ids="3"
dataset="PASTIS24"
command="python -u train_and_eval/segmentation_training_transf.py --config_file configs/PASTIS24/TSViT_fold5.yaml --gpu_ids ${gpu_ids}"
log="./runlog/${time}_data_${dataset}_5.out"
echo $command > $log
echo "use multiattention in space trensformer" > $log # 2023-5-15 07:37:00
nohup $command >> $log 2>&1 &