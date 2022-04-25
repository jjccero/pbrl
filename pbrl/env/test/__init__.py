from gym.envs.registration import register

register(
    id='RnnTest-v0',
    entry_point='pbrl.env.test.rnn:RnnTest',
    max_episode_steps=100,
    reward_threshold=99.0
)
