
## Run Glue Task on qqp dataset
export TASK_NAME=qqp
export WANDB_PROJECT=qqp_glue_3_epochs
python run_glue.py   --model_name_or_path bert-base-uncased   --task_name $TASK_NAME   --do_train   --do_eval   --do_predict --max_seq_length 128   --per_device_train_batch_size 8 --gradient_accumulation_steps 4   --learning_rate 2e-5   --num_train_epochs 3   --output_dir qqp_dataset/qqp_3_epochs/ --overwrite_output_dir --save_steps 40000 --evaluation_strategy steps --eval_steps 5000 --fp16



## Run Glue Task on Orcas dataset

export WANDB_PROJECT=orcas_glue
python run_glue.py   --train_file orcas_dataset/train.csv   --validation_file orcas_dataset/validation.csv --test_file orcas_dataset/test.csv  --model_name_or_path bert-base-uncased   --do_train   --do_eval --do_predict  --max_seq_length 128   --per_device_train_batch_size 16   --gradient_accumulation_steps 2  --learning_rate 2e-5   --num_train_epochs 3   --output_dir orcas_dataset/train_output --overwrite_output_dir --save_steps=50000 --evaluation_strategy steps --eval_steps 40000 --fp16


