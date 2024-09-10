home_dir=$HOME #rerankersがあるディレクトリ

export PYTHONPATH={$home_dir}/rerankers:$PYTHONPATH

data_dir={$home_dir}/rerankers/data
task=bioasq
prefix=default
topk=100
model_name=castorini/monot5-base-msmarco-10k
first_stage=bm25

python driver/rerank.py \
 --task ${task} --rewrite ${prefix} \
 --data_dir ${data_dir} --topk 100 --first_stage $first_stage \
 --model_name $model_name