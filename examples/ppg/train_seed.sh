env='Ant-v3'

for seed in 0 1 2 3 4
do
    python train.py --env ${env} --obs_norm --reward_norm --seed ${seed} --timestep 3072000
done
