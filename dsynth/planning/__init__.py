from dsynth.planning.solve import *
from dsynth.planning.solvers import *

MP_SOLUTIONS = {
    "MoveFromBoardToBoardStaticEnv": solve_fetch_static_from_board_to_board,
    "MoveFromBoardToBoardStaticOneProdEnv": solve_fetch_static_from_board_to_board,
    "MoveFromBoardToBoardEnv": solve_fetch_move_from_board_to_board,
    "PickToCartStatiсOneProdEnv": solve_fetch_pick_to_basket_static 
}