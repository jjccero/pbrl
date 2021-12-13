env='Walker2d-v3'

for seed in 0 1 2
do
    python train.py --env ${env} --obs_norm --reward_norm --seed ${seed}
done
