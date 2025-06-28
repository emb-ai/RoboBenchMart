import numpy as np
np.set_printoptions(suppress=True)

from dsynth.envs import *

from dsynth.planning.motionplanner import (
    FetchMotionPlanningSapienSolver
)
from dsynth.scene_gen.utils import find_paths

def solve_fetch_nav_go_to_zone(env: NavMoveToZoneEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.07
    env = env.unwrapped

    start_cell = env.agent.base_link.pose.sp.p[:2] // CELL_SIZE
    start_cell = (int(start_cell[0]), int(start_cell[1]))

    min_dist_to_shelf = np.inf
    for target_cell, final_view_direction in zip(env.target_cells[0], env.target_directions[0]):
        paths = find_paths(env.room[0], start_cell, 
                           (int(target_cell[0]), int(target_cell[1])))

        # if there are more than one path to shelf
        shortest_path_length = np.inf 
        for path in paths:
            if len(path) < shortest_path_length:
                shortest_path_length = len(path)
                shortest_path_to_shelf = path

        if shortest_path_length < min_dist_to_shelf:
            min_dist_to_shelf = shortest_path_length
            target_cell_closest = target_cell
            target_view_direction = final_view_direction
            target_path = shortest_path_to_shelf
    
    # -------------------------------------------------------------------------- #
    # Drive
    # -------------------------------------------------------------------------- #
    
    for i_step, cur_cell in enumerate(target_path):
        drive_pos = np.array([
            cur_cell[0] * CELL_SIZE + CELL_SIZE / 2,
            cur_cell[1] * CELL_SIZE + CELL_SIZE / 2,
            0.
        ])

        if i_step == 0 and min_dist_to_shelf == 1:
            res = planner.drive_base(target_view_vec=target_view_direction)
        elif i_step == min_dist_to_shelf - 1:
            res = planner.drive_base(target_pos=drive_pos, target_view_vec=target_view_direction)
        elif i_step == 0:
            continue
        else:
            res = planner.drive_base(target_pos=drive_pos)
        planner.planner.update_from_simulation()
    
    return res



