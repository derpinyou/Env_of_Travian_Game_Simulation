import tqdm
import numpy as np

from travian_env import building_dict, building_names, res_names, village_info_dict, \
actions_corr, requirement_dict, npc_deltas, levels, max_levels_dict, TravianEnv
from itertools import chain

for mod in ['culture accum', 'res growth', 'pop total']:
  print('----MOD IS ' + mod + ' now:')
  for boost in [1, 2, 3]:
    print('boost is equal ' + str(boost) + ' now:')
    for g in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
      env = TravianEnv(village_info_dict, building_dict, g, 5184000,
                      requirement_dict, max_levels_dict, npc_deltas,
                      actions_corr, res_names, mod, boost)
      rewards_storage = []
      env.reset()
      for i in range(100):
          while env.current_time <= 5184000:
              while not env.is_available_and_rr(50)[0] and env.current_time <= 5184000:
                w = np.random.choice(list(chain(range(49), range(778, 826))))
                env.step(w)
              while env.gold >= 3 and env.current_time <= 5184000:
                w = np.random.choice(range(778 * len(env.villages_are_available) - 1))
                env.step(w)
              w = np.random.choice(list(chain(range(49), range(778, 826))))
              env.step(w)
          rewards_storage.append(env.Total_r)
          env.reset()
      del env
      print(sum(rewards_storage)/100, g)
