from gym.envs.registration import register

register(
    id='jointcontrol-v0',
    entry_point='gym_jointcontrol.envs:jointcontrol_env',
)