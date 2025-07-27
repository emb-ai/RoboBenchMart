from dsynth.planning.solve import *
from dsynth.planning.solvers import *

MP_SOLUTIONS = {
    "MoveFromBoardToBoardStaticEnv": solve_fetch_static_from_board_to_board,
    "MoveFromBoardToBoardStaticOneProdEnv": solve_fetch_static_from_board_to_board,
    "MoveFromBoardToBoardEnv": solve_fetch_move_from_board_to_board,
    "PickToBasketStaticSpriteEnv": solve_fetch_pick_to_basket_one_prod,
    "PickToBasketSpriteEnv": solve_fetch_pick_to_basket_one_prod,
    "NavMoveToZoneEnv": solve_fetch_nav_go_to_zone,
    "OpenDoorFridgeEnv": solve_fetch_open_door_showcase,

    "PickToBasketContNiveaEnv": solve_fetch_pick_to_basket_cont_one_prod,
    "PickToBasketContStarsEnv": solve_fetch_pick_to_basket_cont_one_prod,
    "PickToBasketContFantaEnv": solve_fetch_pick_to_basket_cont_one_prod,

    "MoveFromBoardToBoardVanishContEnv": solve_fetch_move_to_board_cont_one_prod,
    "MoveFromBoardToBoardNestleContEnv": solve_fetch_move_to_board_cont_one_prod,
    "MoveFromBoardToBoardDuffContEnv": solve_fetch_move_to_board_cont_one_prod,

    "PickFromFloorSlamContEnv": solve_fetch_pick_from_floor_cont,
    "PickFromFloorBeansContEnv": solve_fetch_pick_from_floor_cont,

    "OpenDoorShowcaseContEnv": solve_fetch_open_door_showcase_cont,
    "OpenDoorFridgeContEnv": solve_fetch_open_door_fridge_cont,

    "CloseDoorShowcaseContEnv": solve_fetch_close_door_showcase_cont
}