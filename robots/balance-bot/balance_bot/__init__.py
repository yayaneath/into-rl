from gym.envs.registration import register

register(id='balancebot-v0',
         entry_point='balance_bot.envs:BalancebotEnv',
         max_episode_steps=20000, # Needed for tfagents
)

register(id='balancebotdisc-v0',
         entry_point='balance_bot.envs:BalancebotDiscEnv',
         max_episode_steps=20000, # Needed for tfagents
)