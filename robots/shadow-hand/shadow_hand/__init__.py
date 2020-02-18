from gym.envs.registration import register

register(id='shadowhand-v0',
         entry_point='shadow_hand.envs:ShadowhandEnv',
         max_episode_steps=20000, # Needed for tfagents
)