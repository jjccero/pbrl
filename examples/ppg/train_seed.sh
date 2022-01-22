for seed in 0 100 200 300 400
do
    for env in 'Ant-v3' 'Hopper-v3' 'Walker2d-v3' 'HalfCheetah-v3' 'Swimmer-v3' 'Humanoid-v3'
    do
        python train.py --env ${env} --obs_norm --reward_norm --seed ${seed} --timestep 1024000
    done
done
