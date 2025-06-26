from dsynth.planning.solve import *
from dsynth.planning.solvers import *

MP_SOLUTIONS = {
    "MoveFromBoardToBoardStaticEnv": solve_fetch_static_from_board_to_board,
    "MoveFromBoardToBoardStaticOneProdEnv": solve_fetch_static_from_board_to_board,
    "MoveFromBoardToBoardEnv": solve_fetch_move_from_board_to_board,
    "PickToBasketStaticSpriteEnv": solve_fetch_pick_to_basket_static_one_prod,
    "PickToBasketSpriteEnv": solve_fetch_pick_to_basket_one_prod,
    "NavMoveToZoneEnv": solve_fetch_nav_go_to_zone
}