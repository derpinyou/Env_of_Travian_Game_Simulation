import gym
import pandas as pd

building_names = ['academy', 'bakery', 'barracks', 'brickyard', 'clay', 'crop', 'embassy', 'grainmill', 'granary',
                  'iron', 'ironfoundry', 'main', 'marketplace', 'rally point', 'residence', 'sawmill', 'stable', 'smithy',
                  'townhall', 'warehouse', 'wood']
res_names = ['wood', 'clay', 'iron', 'crop']
building_dict = {}
for name in building_names:
    text_file = open(name + ".txt", "r")
    lines = text_file.readlines()
    df = pd.read_csv(name + '.txt', delimiter="\t", header=None).dropna(how='all', axis=1).dropna(how='all', axis=0)
    if len(df.columns) == 9:
        df.columns = ['level', 'wood', 'clay', 'iron', 'crop', 'res_sum', 'time', 'pop', 'culture']
    else:
        df.columns = ['level', 'wood', 'clay', 'iron', 'crop', 'res_sum', 'time', 'pop', 'culture', 'special']
    building_dict[name] = df.set_index('level').T.to_dict()

village_info_dict = {'village0':          {'crop': [2, 2, 2, 2, 2, 2], 'iron': [1, 1, 1, 1],
                                           'wood': [1, 1, 1, 1], 'clay': [1, 1, 1, 1], 'grainmill': [0],
                                           'sawmill': [0], 'ironfoundry': [0], 'brickyard': [0], 'bakery': [0],
                                           'main': [1], 'granary': [1, 0, 0, 0, 0, 0], 'warehouse': [1, 0, 0, 0, 0, 0],
                                           'marketplace': [0], 'barracks': [0], 'embassy': [0], 'townhall': [0],
                                           'residence': [0], 'smithy': [0], 'stable': [0], 'academy': [0], 'rally point': [0],
                                           'res_wood': [800], 'res_clay': [800], 'res_iron': [800],
                                           'res_crop': [800], 'time_remaining': [0], 'culture_total': [0],
                                           'pop': [32], 'culture': [22], 'waiting_for': 'nothing', 'merchants': [0],
                                           'merchants_time_remaining': [0], 'merchants_waiting_for': [0]},

                     'village1':          {'crop': [1, 1, 1, 1, 1, 1], 'iron': [1, 1, 1, 1],
                                           'wood': [1, 1, 1, 1], 'clay': [1, 1, 1, 1], 'grainmill': [0],
                                           'sawmill': [0], 'ironfoundry': [0], 'brickyard': [0], 'bakery': [0],
                                           'main': [1], 'granary': [1, 0, 0, 0, 0, 0], 'warehouse': [1, 0, 0, 0, 0, 0],
                                           'marketplace': [0], 'barracks': [0], 'embassy': [0], 'townhall': [0],
                                           'residence': [0], 'smithy': [0], 'stable': [0], 'academy': [0], 'rally point': [0],
                                           'res_wood': [0], 'res_clay': [0], 'res_iron': [0],
                                           'res_crop': [0], 'time_remaining': [0], 'culture_total': [0],
                                           'pop': [28], 'culture': [18], 'waiting_for': 'nothing', 'merchants': [0],
                                           'merchants_time_remaining': [0], 'merchants_waiting_for': [0]}
                     }
actions_corr = {'wood': list(range(4)), 'clay': list(range(4, 8)), 'iron': list(range(8, 12)),
                'crop': list(range(12, 18)), 'sawmill': [18], 'brickyard': [19], 'ironfoundry': [20],
                'grainmill': [21], 'bakery': [22], 'main': [23], 'warehouse': list(range(24, 30)),
                'granary': list(range(30, 36)), 'marketplace': [36],
                'barracks': [37], 'embassy': [38], 'townhall': [39], 'residence': [40],
                'smithy': [41], 'stable': [42], 'academy': [43], 'rally point': [44]}

requirement_dict = {'grainmill': {'crop': 5},
                    'sawmill': {'wood': 10, 'main': 5},
                    'ironfoundry': {'iron': 10, 'main': 5},
                    'brickyard': {'clay': 10, 'main': 5},
                    'granary': {'main': 1},
                    'warehouse': {'main': 1},
                    'marketplace': {'main': 3},
                    'barracks': {'main': 3, 'rally point': 1},
                    'embassy': {'main': 1},
                    'townhall': {'main': 10, 'academy': 10},
                    'residence': {'main': 5},
                    'smithy': {'main': 3, 'academy': 3},
                    'academy': {'main': 3, 'barracks': 3}}

npc_deltas = [-5000, -1000, -300, -100, 0,
              100, 300, 1000, 5000]

levels = [20, 5, 20, 5, 21, 21, 20, 5, 20, 20, 5, 20, 20, 20, 20, 5, 20, 20, 20, 20, 20]

max_levels_dict = dict(zip(building_names, levels))

gold = 100
ban = 2592000

building_dict['grainmill'].update({0 : {'special': 100}})
building_dict['bakery'].update({0 : {'special': 100}})
building_dict['sawmill'].update({0 : {'special': 100}})
building_dict['ironfoundry'].update({0 : {'special': 100}})
building_dict['brickyard'].update({0 : {'special': 100}})


class TravianEnv(gym.Env):

    def __init__(self, x, buildings_info, gold, ban, requirement_dict, max_levels_dict,
                 npc_deltas, act_corr, r_n, mod, x_boost):
        self.ban = ban
        self.x_boost = x_boost
        self.X = x
        self.village_n = len(x)
        self.buildings_info = buildings_info
        self.gold = gold
        self.action_space = gym.spaces.Discrete(778 * self.village_n)
        self.current_time = 0
        self.actions_corr = act_corr
        self.requirements = requirement_dict
        self.Total_r = 0
        self.max_levels_dict = max_levels_dict
        self.npc_deltas = npc_deltas
        self.villages_are_available = [0]
        self.granary_capacities = {i: self.current_capacity_and_boost(i)['granary'] for i in range(self.village_n)}
        self.storage_capacities = {i: self.current_capacity_and_boost(i)['warehouse'] for i in range(self.village_n)}
        self.boost = {i: self.current_capacity_and_boost(i)['main'] for i in range(self.village_n)}
        self.res_growths = {i: self.res_growth(i) for i in range(self.village_n)}
        self.Total_culture = sum([self.count_culture_per_day(i) for i in self.villages_are_available])
        self.Total_culture_accum = 0
        self.res_names = r_n
        self.mod = mod


    def count_culture_per_day(self, vil_n):
        """
        count culture per day for a village
        :param self:
        :param vil_n: villages num
        :return: total culture among all the villages
        """
        total_cult = 0
        for building in self.actions_corr:
            cult = sum([self.buildings_info[building][level]['culture'] for
                        level in self.X['village' + str(vil_n)][building] if level != 0])
            total_cult += cult
        return total_cult / (3600 * 24)

    def current_capacity_and_boost(self, vil_n):
        """
        count capacities of granaries and warehouses and a boost from main building
        :param self:
        :param vil_n: villages num
        :return: list [granary capacity, warehouse capacity, main boost]
        """
        dict_of_boosts = {}
        for building in ['granary', 'warehouse', 'main']:
            dict_of_boosts[building] = sum([self.buildings_info[building][level]['special'] for
                                            level in self.X['village' + str(vil_n)][building] if level != 0])
        return dict_of_boosts

    def res_growth(self, vil_n):
        """
        count resources production per time
        :param self:
        :param vil_n: villages num
        :return: list [wood prod, clay prod, iron prod, crop prod - population]
        """
        dict_of_res_prods = {}
        for building in ['wood', 'clay', 'iron', 'crop']:
            dict_of_res_prods[building] = self.x_boost * sum([self.buildings_info[building][level]['special'] for
                                               level in self.X['village' + str(vil_n)][building] if level != 0])
        bakery_lvl = self.X['village' + str(vil_n)]['bakery'][0]
        grainmill_lvl = self.X['village' + str(vil_n)]['grainmill'][0]
        sawmill_lvl = self.X['village' + str(vil_n)]['sawmill'][0]
        brickyard_lvl = self.X['village' + str(vil_n)]['brickyard'][0]
        ironfoundry_lvl = self.X['village' + str(vil_n)]['ironfoundry'][0]

        grainmill_boost = dict_of_res_prods['crop'] * (int(
            str(self.buildings_info['grainmill'][grainmill_lvl]['special']).split('%')[0].split('.')[0]) - 100) / 100
        dict_of_res_prods['crop'] *= int(
            str(self.buildings_info['bakery'][bakery_lvl]['special']).split('%')[0].split('.')[0]) / 100
        dict_of_res_prods['crop'] += grainmill_boost

        dict_of_res_prods['wood'] *= int(
            str(self.buildings_info['sawmill'][sawmill_lvl]['special']).split('%')[0].split('.')[0]) / 100
        dict_of_res_prods['clay'] *= int(
            str(self.buildings_info['brickyard'][brickyard_lvl]['special']).split('%')[0].split('.')[0]) / 100
        dict_of_res_prods['iron'] *= int(
            str(self.buildings_info['ironfoundry'][ironfoundry_lvl]['special']).split('%')[0].split('.')[0]) / 100

        dict_of_res_prods['crop'] -= self.X['village' + str(vil_n)]['pop'][0]
        return dict_of_res_prods

    def count_time_pace(self):
        """
        tune how much time an agent 'waits' if a corresponding action is chosen
        :param self:
        :return: min(time remaining among all the villages; time remaining for any one more available building)
        """

        min_time_for_next = []
        for village in self.villages_are_available:
            for building in building_names:
                lvls = self.X['village' + str(village)][building]
                next_lvls = [lvl + 1 for lvl in lvls if lvl < max(list(self.buildings_info[building].keys()))]
                for one_lvl in next_lvls:
                    req_res = [self.buildings_info[building][one_lvl][res] for res in self.res_names]
                    capacities = [self.granary_capacities[village], self.storage_capacities[village],
                                  self.storage_capacities[village], self.storage_capacities[village]]
                    helpful_list = [capacities[i] - req_res[i] for i in range(len(capacities))]
                    if len([x for x in helpful_list if x > 0]) != 4:
                        next_lvls = [y for y in next_lvls if y != one_lvl]
                if len(next_lvls) != 0:
                    list_of_dicts = [self.buildings_info[building][n_lvl] for n_lvl in next_lvls]
                    list_of_lists_of_res = [[x[res] for res in self.res_names] for x in list_of_dicts]
                    current_res = [self.X['village' + str(village)]['res_' + res][0] for res in self.res_names]
                    deltas = [[need_res - cur_res for need_res, cur_res
                               in zip(list_of_lists_of_res[k], current_res)] for k in range(len(list_of_lists_of_res))]
                    to_count_speed = self.res_growths[village].values()
                    times = [[3600 * i / j for i, j in zip(delta_list, to_count_speed)] for delta_list in deltas]
                    min_time = min([max(list_of_t) for list_of_t in times])
                    min_time_for_next.append(min_time)

            current_res = [self.X['village' + str(village)]['res_' + res][0] for res in self.res_names]

            if self.X['village' + str(village)]['merchants'][0] >= 3:
                deltas = [[need_res - cur_res for need_res, cur_res
                           in zip([750, 750, 750, 750], current_res)]]
                to_count_speed = self.res_growths[village].values()
                times = [[3600 * i / j for i, j in zip(delta_list, to_count_speed)] for delta_list in deltas]

                min_time = min([max(list_of_t) if max(list_of_t) > 0 else -1000 for list_of_t in times])
                min_time_for_next.append(min_time)

            if self.X['village' + str(village)]['embassy'][0] >= 10:
                deltas = [[need_res - cur_res for need_res, cur_res
                           in zip([4500, 4500, 4500, 4500], current_res)]]
                to_count_speed = self.res_growths[village].values()
                times = [[3600 * i / j for i, j in zip(delta_list, to_count_speed)] for delta_list in deltas]
                min_time = min([max(list_of_t) if max(list_of_t) > 0 else -1000 for list_of_t in times])
                min_time_for_next.append(min_time)

            min_time_for_next.append(self.X['village' + str(village)]['time_remaining'][0])
            min_time_for_next.append(self.X['village' + str(village)]['merchants_time_remaining'][0])
        if len([z for z in min_time_for_next if z > 0]) != 0:
            return min([x for x in min_time_for_next if x > 0])
        else:
            return 0

    def is_available_and_rr(self, action_):
        """
        indicates whether chosen action is available or not, calculates its resources and time costs
        :param self:
        :param action_:
        :return: is_available boolean, res costs, time costs, building_name indicator if construction has started
        """
        excess_ = action_ % 778
        integer_ = action_ // 778
        is_constructing = self.X['village' + str(integer_)]['time_remaining'][0] != 0
        available = 0
        res_costs = 0
        time_costs = 0
        build_name = ' '
        is_village_based = self.villages_are_available.count(integer_) != 0
        if excess_ <= 44 and not is_constructing and is_village_based:  # construct something option
            build_name = [k for k, v in self.actions_corr.items() if v.count(excess_) != 0][0]
            build_position = self.actions_corr[build_name].index(excess_)
            build_current_level = self.X['village' + str(integer_)][build_name][build_position]

            if build_current_level < max(list(self.buildings_info[build_name].keys())):
                h, m, s = self.buildings_info[build_name][build_current_level + 1]['time'].split(':')
                time_costs = (3600 * int(h) + 60 * int(m) + int(s)) * self.boost[integer_] / 100

                res_costs = [self.buildings_info[build_name][build_current_level + 1][x] for x in self.res_names]
                res_current = [self.X['village' + str(integer_)]['res_' + x][0] for x in self.res_names]

                res_deltas = [k - v for k, v in zip(res_current, res_costs)]

                new_pop = self.buildings_info[build_name][build_current_level + 1]['pop']
                good_balance = new_pop < self.res_growths[integer_]['crop']

                if build_name in list(self.requirements.keys()):
                    how_much_req = len(self.requirements[build_name].items())
                    is_ok = 0

                    for req in self.requirements[build_name].items():
                        lvl_list = self.X['village' + str(integer_)][req[0]]
                        if len([x for x in lvl_list if x >= req[1]]) > 0:
                            is_ok += 1

                    requirements_are_good = how_much_req == is_ok
                else:
                    requirements_are_good = True

                fine_level = build_current_level < self.max_levels_dict[build_name]

                if len([y for y in res_deltas if
                        y >= 0]) == 4 and good_balance and requirements_are_good and fine_level:
                    available = 1
                else:
                    available = 0

        elif excess_ == 45:  # wait option
            if integer_ == 0 and self.count_time_pace() != 0:
                available = 1
            res_costs = 0
            time_costs = self.count_time_pace()
            build_name = 'wait'

        elif excess_ == 46:  # fast construction option
            if self.X['village' + str(integer_)]['time_remaining'][0] != 0:
                available = 1
                if self.X['village' + str(integer_)]['time_remaining'][0] <= 7200 and self.gold >= 1:
                    res_costs = 1
                elif self.X['village' + str(integer_)]['time_remaining'][0] > 7200 and self.gold >= 2:
                    res_costs = 2
                else:
                    available = 0
            time_costs = 0
            build_name = 'donat_1'

        elif excess_ == 47:  # hire merchants option
            res_costs = [4500, 4500, 4500, 4500]
            time_costs = 9000
            build_name = 'merch'

            merch_not_constructing = self.X['village' + str(integer_)]['merchants_time_remaining'][0] == 0
            res_current = [self.X['village' + str(integer_)]['res_' + x][0] for x in self.res_names]
            enough_res = len([x for x in res_current if x >= 4500]) == 4

            if self.X['village' + str(integer_)]['embassy'][0] >= 10 and merch_not_constructing and enough_res:
                available = 1

        elif excess_ == 48 and is_village_based:  # new village option
            enough_merch = self.X['village' + str(integer_)]['merchants'][0] > 2
            res_current = [self.X['village' + str(integer_)]['res_' + x][0] for x in self.res_names]
            enough_res = len([x for x in res_current if x >= 750]) == 4
            no_more_than_two = len(self.villages_are_available) == 1
            if enough_merch and enough_res and no_more_than_two:
                available = 1
                res_costs = [750, 750, 750, 750]
                time_costs = 0
                build_name = 'new_vil'
        else:
            if is_village_based and 49 <= excess_ <= 777:
                is_market_built = self.X['village' + str(integer_)]['marketplace'][0] != 0

                new_excess = excess_ - 49
                wood_to_trade = self.npc_deltas[new_excess % 9]
                iron_to_trade = self.npc_deltas[new_excess // 9 - 9 * (new_excess // (9 ** 2))]
                clay_to_trade = self.npc_deltas[new_excess // (9 ** 2)]
                crop_to_trade = -(wood_to_trade + iron_to_trade + clay_to_trade)

                wood_after_trade = self.X['village' + str(integer_)]['res_wood'][0] + wood_to_trade
                iron_after_trade = self.X['village' + str(integer_)]['res_iron'][0] + iron_to_trade
                clay_after_trade = self.X['village' + str(integer_)]['res_clay'][0] + clay_to_trade
                crop_after_trade = self.X['village' + str(integer_)]['res_crop'][0] + crop_to_trade

                not_negative = wood_after_trade >= 0 and iron_after_trade >= 0 and \
                               clay_after_trade >= 0 and crop_after_trade >= 0

                not_exceed_crop = self.granary_capacities[integer_] >= crop_after_trade
                not_exceed_others = self.storage_capacities[integer_] >= iron_after_trade and \
                                    self.storage_capacities[integer_] >= wood_after_trade and \
                                    self.storage_capacities[integer_] >= clay_after_trade

                if is_market_built and not_negative and not_exceed_crop and not_exceed_others and self.gold >= 3:
                    available = 1

                time_costs = 0
                build_name = 'npc swap'

                res_costs = {'wood': wood_after_trade, 'iron': iron_after_trade,
                             'clay': clay_after_trade, 'crop': crop_after_trade, 'gold': 3}

        return available == 1, res_costs, time_costs / self.x_boost, build_name

    def step(self, action_):
        """
        change state in response to action if available
        :param self:
        :param action_:
        :return: obs, reward, is_done, metadata
        """

        done = False
        reward = 0
        excess_ = action_ % 778
        integer_ = action_ // 778
        if not self.is_available_and_rr(action_)[0]:
            return 'UNAVAILABLE', reward, done, {}
        if self.is_available_and_rr(action_)[0]:
            if excess_ <= 44:
                build_name = self.is_available_and_rr(action_)[3]
                res_costs = {self.res_names[i]: self.is_available_and_rr(action_)[1][i] for i in range(4)}
                time_costs = self.is_available_and_rr(action_)[2]

                for res in self.res_names:
                    self.X['village' + str(integer_)]['res_' + res][0] -= res_costs[res]

                build_position = self.actions_corr[build_name].index(excess_)
                self.X['village' + str(integer_)]['time_remaining'][0] = time_costs
                self.X['village' + str(integer_)]['waiting_for'] = [build_name, build_position]

            elif excess_ == 45:

                time_step = self.count_time_pace()
                self.current_time += time_step

                for vil in self.villages_are_available:
                    if time_step >= self.X['village' + str(vil)]['time_remaining'][0]:
                        if self.X['village' + str(vil)]['waiting_for'] != 'nothing':
                            build_name = self.X['village' + str(vil)]['waiting_for'][0]
                            build_position = self.X['village' + str(vil)]['waiting_for'][1]
                            self.X['village' + str(vil)][build_name][build_position] += 1
                            level = self.X['village' + str(vil)][build_name][build_position]
                            if level != 1:
                                previous_cult = self.buildings_info[build_name][level - 1]['culture']
                            else:
                                previous_cult = 0
                            current_cult = self.buildings_info[build_name][level]['culture']

                            cult_delta = current_cult - previous_cult

                            self.X['village' + str(vil)]['culture'][0] += cult_delta
                            self.X['village' + str(vil)]['pop'][0] += self.buildings_info[build_name][level]['pop']

                            if self.mod == 'pop total':
                                reward += self.buildings_info[build_name][level]['pop']

                        self.X['village' + str(vil)]['time_remaining'][0] = 0
                        self.X['village' + str(vil)]['waiting_for'] = 'nothing'
                    else:
                        self.X['village' + str(vil)]['time_remaining'][0] -= time_step

                    if time_step >= self.X['village' + str(vil)]['merchants_time_remaining'][0]:
                        if self.X['village' + str(vil)]['merchants_waiting_for'][0] != 0:
                            self.X['village' + str(vil)]['merchants_waiting_for'][0] -= 1
                            self.X['village' + str(vil)]['merchants'][0] += 1
                        self.X['village' + str(vil)]['merchants_time_remaining'][0] = 0
                    else:
                        self.X['village' + str(vil)]['merchants_time_remaining'][0] -= time_step

                    for rs in self.res_names:
                        to_add_res = self.res_growths[vil][rs] * time_step / 3600
                        lim = self.X['village' + str(vil)]['res_' + rs][0] + to_add_res
                        if rs == 'crop':
                            self.X['village' + str(vil)]['res_' + rs][0] = min(lim, self.granary_capacities[vil])
                        else:
                            self.X['village' + str(vil)]['res_' + rs][0] = min(lim, self.storage_capacities[vil])

                self.Total_culture_accum += self.Total_culture * time_step
                if self.mod == 'culture accum':
                    reward = self.Total_culture * time_step
            elif excess_ == 46:
                self.gold -= self.is_available_and_rr(action_)[1]
                self.X['village' + str(integer_)]['time_remaining'][0] = 0
                build_name = self.X['village' + str(integer_)]['waiting_for'][0]
                build_position = self.X['village' + str(integer_)]['waiting_for'][1]
                self.X['village' + str(integer_)][build_name][build_position] += 1
                self.X['village' + str(integer_)]['waiting_for'] = 'nothing'

            elif excess_ == 47:

                res_costs = {self.res_names[i]: self.is_available_and_rr(action_)[1][i] for i in range(4)}

                for res in self.res_names:
                    self.X['village' + str(integer_)]['res_' + res][0] -= res_costs[res]

                self.X['village' + str(integer_)]['merchants_waiting_for'][0] += 1
                self.X['village' + str(integer_)]['merchants_time_remaining'][0] += 9000

            elif excess_ == 48:

                res_costs = {self.res_names[i]: self.is_available_and_rr(action_)[1][i] for i in range(4)}

                for res in self.res_names:
                    self.X['village' + str(integer_)]['res_' + res][0] -= res_costs[res]

                self.villages_are_available.append(1 + max(self.villages_are_available))
                self.X['village' + str(integer_)]['merchants'][0] -= 3

            elif excess_ >= 48:
                self.gold -= self.is_available_and_rr(action_)[1]['gold']
                res_costs = self.is_available_and_rr(action_)[1]
                res_costs.pop('gold')

                for res in self.res_names:
                    self.X['village' + str(integer_)]['res_' + res][0] = res_costs[res]

        total_res_growth_prev = sum([sum(list(self.res_growths[vil].values())) for vil in self.villages_are_available])

        self.granary_capacities = {i: self.current_capacity_and_boost(i)['granary'] for i in range(self.village_n)}
        self.storage_capacities = {i: self.current_capacity_and_boost(i)['warehouse'] for i in range(self.village_n)}
        self.boost = {i: self.current_capacity_and_boost(i)['main'] for i in range(self.village_n)}
        self.res_growths = {i: self.res_growth(i) for i in range(self.village_n)}

        total_res_growth_now = sum([sum(list(self.res_growths[vil].values())) for vil in self.villages_are_available])

        if self.mod == 'res growth':
            reward = total_res_growth_now - total_res_growth_prev

        self.Total_r += reward
        self.Total_culture = sum([self.count_culture_per_day(i) for i in self.villages_are_available])

        obs = self.X, self.gold, self.storage_capacities, self.granary_capacities, \
              self.boost, self.res_growths, self.current_time, self.Total_r

        if self.current_time < 5184000:
            done = False
        else:
            done = True
        return obs, reward, done, {}

    def reset(self):
        """
        reset a session to its initial state
        :param self:
        :return: initial state
        """

        self.X = {'village0': {'crop': [2, 2, 2, 2, 2, 2], 'iron': [1, 1, 1, 1],
                               'wood': [1, 1, 1, 1], 'clay': [1, 1, 1, 1], 'grainmill': [0],
                               'sawmill': [0], 'ironfoundry': [0], 'brickyard': [0], 'bakery': [0],
                               'main': [1], 'granary': [1, 0, 0, 0, 0, 0], 'warehouse': [1, 0, 0, 0, 0, 0],
                               'marketplace': [0], 'barracks': [0], 'embassy': [0], 'townhall': [0],
                               'residence': [0], 'smithy': [0], 'stable': [0], 'academy': [0], 'rally point': [0],
                               'res_wood': [800], 'res_clay': [800], 'res_iron': [800],
                               'res_crop': [800], 'time_remaining': [0], 'culture_total': [0],
                               'pop': [28], 'culture': [18], 'waiting_for': 'nothing', 'merchants': [0],
                               'merchants_time_remaining': [0], 'merchants_waiting_for': [0]},

                  'village1': {'crop': [1, 1, 1, 1, 1, 1], 'iron': [1, 1, 1, 1],
                               'wood': [1, 1, 1, 1], 'clay': [1, 1, 1, 1], 'grainmill': [0],
                               'sawmill': [0], 'ironfoundry': [0], 'brickyard': [0], 'bakery': [0],
                               'main': [1], 'granary': [1, 0, 0, 0, 0, 0], 'warehouse': [1, 0, 0, 0, 0, 0],
                               'marketplace': [0], 'barracks': [0], 'embassy': [0], 'townhall': [0],
                               'residence': [0], 'smithy': [0], 'stable': [0], 'academy': [0], 'rally point': [0],
                               'res_wood': [0], 'res_clay': [0], 'res_iron': [0],
                               'res_crop': [0], 'time_remaining': [0], 'culture_total': [0],
                               'pop': [28], 'culture': [18], 'waiting_for': 'nothing', 'merchants': [0],
                               'merchants_time_remaining': [0], 'merchants_waiting_for': [0]}
                  }

        self.gold = gold
        self.current_time = 0
        self.Total_r = 0
        self.villages_are_available = [0]
        self.granary_capacities = {i: self.current_capacity_and_boost(i)['granary'] for i in range(self.village_n)}
        self.storage_capacities = {i: self.current_capacity_and_boost(i)['warehouse'] for i in range(self.village_n)}
        self.boost = {i: self.current_capacity_and_boost(i)['main'] for i in range(self.village_n)}
        self.res_growths = {i: self.res_growth(i) for i in range(self.village_n)}
        self.Total_culture = sum([self.count_culture_per_day(i) for i in self.villages_are_available])
        self.Total_culture_accum = 0

        return [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 800, 800, 800, 800, 0, 0, 28, 18, 0, 0, 0,

                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 18, 0, 0, 0,
                self.res_growths[0]['wood'], self.res_growths[0]['clay'],
                self.res_growths[0]['iron'], self.res_growths[0]['crop'],
                self.res_growths[1]['wood'], self.res_growths[1]['clay'],
                self.res_growths[1]['iron'], self.res_growths[1]['crop'],
                self.granary_capacities[0],
                self.granary_capacities[1], self.storage_capacities[1], self.storage_capacities[0],
                self.boost[0], self.boost[1], self.gold, self.current_time]

    def reset_high(self):
        self.X = {'village0': {'crop': [10, 10, 10, 10, 10, 10], 'iron': [10, 10, 10, 10],
                               'wood': [10, 10, 10, 10], 'clay': [10, 10, 10, 10], 'grainmill': [5],
                               'sawmill': [5], 'ironfoundry': [5], 'brickyard': [5], 'bakery': [5],
                               'main': [10], 'granary': [10, 10, 10, 10, 10, 10], 'warehouse': [10, 10, 10, 10, 10, 10],
                               'marketplace': [10], 'barracks': [10], 'embassy': [10], 'townhall': [10],
                               'residence': [10], 'smithy': [10], 'stable': [10], 'academy': [10], 'rally point': [10],
                               'res_wood': [11800], 'res_clay': [11800], 'res_iron': [11800],
                               'res_crop': [11800], 'time_remaining': [0], 'culture_total': [0],
                               'pop': [28], 'culture': [18], 'waiting_for': 'nothing', 'merchants': [0],
                               'merchants_time_remaining': [0], 'merchants_waiting_for': [0]},

                  'village1': {'crop': [1, 1, 1, 1, 1, 1], 'iron': [1, 1, 1, 1],
                               'wood': [1, 1, 1, 1], 'clay': [1, 1, 1, 1], 'grainmill': [0],
                               'sawmill': [0], 'ironfoundry': [0], 'brickyard': [0], 'bakery': [0],
                               'main': [1], 'granary': [1, 0, 0, 0, 0, 0], 'warehouse': [1, 0, 0, 0, 0, 0],
                               'marketplace': [0], 'barracks': [0], 'embassy': [0], 'townhall': [0],
                               'residence': [0], 'smithy': [0], 'stable': [0], 'academy': [0], 'rally point': [0],
                               'res_wood': [0], 'res_clay': [0], 'res_iron': [0],
                               'res_crop': [0], 'time_remaining': [0], 'culture_total': [0],
                               'pop': [28], 'culture': [18], 'waiting_for': 'nothing', 'merchants': [0],
                               'merchants_time_remaining': [0], 'merchants_waiting_for': [0]}
                  }

        self.gold = gold
        self.current_time = 0
        self.Total_r = 0
        self.villages_are_available = [0]
        self.granary_capacities = {i: self.current_capacity_and_boost(i)['granary'] for i in range(self.village_n)}
        self.storage_capacities = {i: self.current_capacity_and_boost(i)['warehouse'] for i in range(self.village_n)}
        self.boost = {i: self.current_capacity_and_boost(i)['main'] for i in range(self.village_n)}
        self.res_growths = {i: self.res_growth(i) for i in range(self.village_n)}
        self.Total_culture = sum([self.count_culture_per_day(i) for i in self.villages_are_available])
        self.Total_culture_accum = 0

        return 'reset to high performance village'