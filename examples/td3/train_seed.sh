env='Humanoid-v3'

for seed in 0 1 2
do
    python train.py --env ${env} --obs_norm --seed ${seed}
done
