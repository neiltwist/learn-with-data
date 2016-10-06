# coding=utf-8
import time

import numpy as np

NUM_EXPERIMENTS = 100000
NUM_COINS = 1000
NUM_FLIPS = 10


def get_heads(flips):
    return len(filter(lambda x: x >= 0, flips))


def run_experiment(N_coin, N_flip):
    random_coin = int(N_coin * np.random.random())
    c_1 = None
    min_heads = N_flip + 1  # initialise to the maximum possible
    for coin_number in range(N_coin):
        flips = np.array(2 * np.random.random(N_flip) - 1)
        if c_1 is None:
            c_1 = np.copy(flips)
        if coin_number == random_coin:
            c_rand = np.copy(flips)
        heads = get_heads(flips)
        if heads < min_heads:
            min_heads = heads
            min_heads_at = coin_number
            c_min = np.copy(flips)
    # print("c_1 had {0} heads".format(get_heads(c_1)))
    # print("c_rand had {0} heads".format(get_heads(c_rand)))
    # print("c_min had {0} heads at coin {1}".format(get_heads(c_min), min_heads_at))
    # print get_heads(c_1) / float(N_flip), get_heads(c_rand) / float(N_flip), get_heads(c_min) / float(N_flip),\
    #        get_heads(c_1), get_heads(c_rand), get_heads(c_min)
    return get_heads(c_1) / float(N_flip), get_heads(c_rand) / float(N_flip), get_heads(c_min) / float(N_flip)

start_time = time.time()
distribution = [run_experiment(NUM_COINS, NUM_FLIPS) for _ in range(NUM_EXPERIMENTS)]
print("--- %s seconds ---" % (time.time() - start_time))

print("{0} experiments run of {1} coins and {2} flips per coin\n".format(len(distribution), NUM_COINS, NUM_FLIPS))
nu_one = np.mean(map(lambda x: x[0], distribution))
nu_ran = np.mean(map(lambda x: x[1], distribution))
nu_min = np.mean(map(lambda x: x[2], distribution))

print("ν_1={0}\nν_rand={1}\nν_min={2}".format(nu_one, nu_ran, nu_min))
print("--- %s seconds ---" % (time.time() - start_time))
