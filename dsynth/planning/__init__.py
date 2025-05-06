from dsynth.planning.solve import *
from dsynth.planning.solvers import *

MP_SOLUTIONS = {
    "MoveFromBoardToBoardStaticEnv": solve_fetch_static_from_board_to_board,
     "MoveFromBoardToBoardEnv": solve_fetch_move_from_board_to_board 
}