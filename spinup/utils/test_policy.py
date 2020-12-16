import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from spinup.utils.run_utils import set_mujoco
from mujoco_py import MjSimState


def load_policy_and_env(fpath, itr='last', deterministic=False, device=None):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][6:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model_' in x]

        itr = f'{max(saves):06d}' if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = f'{itr:06d}'

    # load the get_action function
    if backend == 'tf1':
        get_action, model = load_tf_policy(fpath, itr, deterministic, device=device)
    else:
        get_action, model = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)

    set_mujoco()
    state = joblib.load(osp.join(fpath, 'state', f'vars_{itr}.pkl'))
    env = state['env']

    # try:
    #     set_mujoco()
    #     state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
    #     env = state['env']
    # except:
    #     env = None

    return env, get_action, model


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    return get_action, model


def load_pytorch_policy(fpath, itr, deterministic=False, device=None):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model_' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)

    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    # make function for producing an action given a single state
    def get_action(x, **parameters):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            action = model.act(x, **parameters)
        return action

    return get_action, model


def get_state(env):
    return env.sim.get_state().flatten()


def set_state(s, env):
    env.reset()
    mj_state = MjSimState.from_flattened(s, env.sim)
    env.sim.set_state(mj_state)
    env.sim.forward()
    return env.env._get_obs()


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, sleep=1e-3,
               log=True, verbose=True, reset_state=None, q_action=None, action_parameters=None, random=False):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    if log:
        logger = EpochLogger()

    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    o = env.reset() if reset_state is None else set_state(reset_state, env)

    action_parameters = {} if action_parameters is None else action_parameters

    while n < num_episodes:
        img = None
        if render:
            img = env.render(mode='rgb_array')
            time.sleep(sleep)

        if ep_len == 0 and q_action is not None:
            a = q_action
        elif random:
            a = env.action_space.sample()
        else:
            a = get_action(o, **action_parameters)

        o_prev = o
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        results = {'img': img, 'a': a, 'r': r, 'd': d, 'score': ep_ret, 't': ep_len, 'o': o_prev}

        yield results

        if d or (ep_len == max_ep_len):

            if log:
                logger.store(EpRet=ep_ret, EpLen=ep_len)

            if verbose:
                print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))

                r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
                o = env.reset() if reset_state is None else set_state(reset_state, env)

            n += 1

    if log:
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--sleep', type=float, default=0.001)
    args = parser.parse_args()
    env, get_action, model = load_policy_and_env(args.fpath,
                                                 args.itr if args.itr >= 0 else 'last',
                                                 args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender), sleep=args.sleep)