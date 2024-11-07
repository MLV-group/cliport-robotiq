"""Data collection script."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

# import os, sys
# ci_build_and_not_headless = False
# try:
#     from cv2.version import ci_build, headless
#     ci_and_not_headless = ci_build and not headless
# except:
#     pass
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_FONTDIR")

RECORD_NUM = 50

def save_debug(act, cmap):
    ''' debugging '''
    pick_pose, place_pose = act['pose0'], act['pose1']
    import matplotlib.pyplot as plt
    import os
    from cliport.utils import utils
    import pybullet as p
    os.makedirs("/home/commonsense/data/cvpr/cliport/test_image", exist_ok=True)
    color = cmap
    color = color.transpose(1,0,2)
    fig, ax = plt.subplots()
    
    p0 = utils.xyz_to_pix(pick_pose[0], np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    p1 = utils.xyz_to_pix(place_pose[0], np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    pick_euler = p.getEulerFromQuaternion(pick_pose[1])
    place_euler = p.getEulerFromQuaternion(place_pose[1])
    p0_theta = pick_euler[2]  # Z축 회전 각도
    p1_theta = place_euler[2]
    
    pick = p0
    place = p1
    
    pick_circle = plt.Circle(pick, 3, color='r', fill=False)
    place_circle = plt.Circle(place, 3, color='g', fill=False)
    plt.imshow(color)
    
    line_len = 30
    pick0 = (pick[0] + line_len/2.0 * np.sin(p0_theta), pick[1] + line_len/2.0 * np.cos(p0_theta))
    pick1  = (pick[0] - line_len/2.0 * np.sin(p0_theta), pick[1] - line_len/2.0 * np.cos(p0_theta))
    ax.plot((pick1[0], pick0[0]), (pick1[1], pick0[1]), color='r', linewidth=2)

    place0 = (place[0] + line_len/2.0 * np.sin(p1_theta), place[1] + line_len/2.0 * np.cos(p1_theta))
    place1  = (place[0] - line_len/2.0 * np.sin(p1_theta), place[1] - line_len/2.0 * np.cos(p1_theta))
    ax.plot((place1[0], place0[0]), (place1[1], place0[1]), color='g', linewidth=2)

    ax.add_patch(pick_circle)
    ax.add_patch(place_circle)
    ax.axis('off')
    plt.savefig(f"/home/commonsense/data/cvpr/cliport/test_image/{random.randint(0, 100)}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    # print(pick_pose)
    # print(utils.xyz_to_pix(pick_pose[0], np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125))
    # print(color.shape)

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset(augment=True)
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record and dataset.n_episodes % (cfg['n']/RECORD_NUM) == 0:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        try:
            for _ in range(task.max_steps):

                act, cmap = agent.act(obs, info)

                save_debug(act, cmap)
                
                episode.append((obs, act, reward, info))
                lang_goal = info['lang_goal']
                obs, reward, done, info = env.step(act)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                if done:
                    break
        except:
            continue
        episode.append((obs, None, reward, info))

        # End video recording
        if record and dataset.n_episodes % (cfg['n']/RECORD_NUM) == 0:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    main()
