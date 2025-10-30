# RoboBenchMart Tasks

We present seven atomic and three composite tasks for evaluating mobile manipulation policies in retail environments.
These tasks can be grouped into two categories: **pick-and-place (PnP)** and **opening/closing**.

### Pick-and-Place Tasks

* **PickToBasket**: pick a specified item and place it in the attached basket  
* **MoveFromBoardToBoard**: move a specified item to the next board  
* **PickFromFloor**: pick an item from the floor and place it on a shelf

### Opening and Closing Tasks

* **Open Showcase** / **Close Showcase**: open or close a specified door of the vertical showcase  
* **Open Fridge** / **Close Fridge**: open or close the ice cream fridge door

Each task consists of two components:

* **ManiSkill environment** – defines the target object (for PnP), success criteria, robot initial position, and wall/ceiling textures  
* **Scene data** – defines layouts, objects present in the scenes, and their arrangement on shelves, which are imported into the ManiSkill environment during episode initialization

Thus, different target objects require different ManiSkill environments, and different item sets require distinct scene data.

Each atomic task is evaluated under the following setups:

* **Train scenes** – same scenes and target objects as used in training  
* **Train scenes with initial pose randomization** – same as training but with a different initial robot pose  
* **Test scenes** – unseen layouts and object arrangements, but seen target objects  
* **Out-of-distribution items** – unseen scenes and unseen target items (for PnP tasks only)

## Atomic PnP Tasks

In pick-and-place (PnP) task we split target pickable items into seen and unseen items during training to test generalization abilities of generalist policies. 

<table>
<tr>
<th>

</th>
<th>PickToBasket</th>
<th>MoveFromBoardToBoard</th>
<th>PickFromFloor</th>
</tr>
<tr>
<td>Train items</td>
<td>

* NiveaBodyMilk
* NestleHoneyStars
* FantaSaborNaranja2L
</td>
<td>

* NestleFitnessChocolateCereals
* DuffBeerCan
* VanishStainRemover
</td>
<td>

* HeinzBeansInARichTomatoSauce
* SlamLuncheonMeat
</td>
</tr>
<tr>
<td>OOD test items</td>
<td>

* NestleFitnessChocolateCereals
* SlamLuncheonMeat
</td>
<td>

* NiveaBodyMilk
* FantaSaborNaranja2L

</td>
<td>

* FantaSaborNaranja2L
* DuffBeerCan
</td>
</tr>
<tr>
<td>#layouts</td>
<td>20</td>
<td>10x3</td>
<td>10</td>
</tr>
<tr>
<td>#trajs</td>
<td>248x3</td>
<td>248x3</td>
<td>248x3</td>
</tr>
</table>

### PickToBasket

**Task Description:**
Approach the shelf and pick up any item with specified name, placing it into the basket attached to the Fetch robot.
The robot is spawned in close proximity to the shelf.


#### Train environments

ManiSkill environments: `PickToBasketContNiveaEnv`, `PickToBasketContStarsEnv`, `PickToBasketContFantaEnv`.

Scene configs: `conf/pick_to_basket_1`, `conf/pick_to_basket_2`.

#### Test environments

ManiSkill environments with OOD target items: `PickToBasketContNestleEnv`, `PickToBasketContSlamEnv`, `PickToBasketContDuffEnv`.

Scene configs: `conf/test_unseen_scenes_pick_to_basket_1`, `conf/test_unseen_scenes_pick_to_basket_2`,
`conf/test_unseen_items_pick_to_basket_1`, `conf/test_unseen_items_pick_to_basket_2`.

### PickFromFloor

**Task Description:**
Approach to the shelf, pick the fallen item and place it on the shelf.
The robot is spawned in close proximity to the shelf. The goal position for the fallen item is its original location on the shelf.


#### Train environments

ManiSkill environments: `PickFromFloorBeansContEnv`, `PickFromFloorSlamContEnv`.

Scene configs: `conf/pick_from_floor_1`, `conf/pick_from_floor_2`.

#### Test environments

ManiSkill environments with OOD target items: `PickFromFloorFantaContEnv`, `PickFromFloorDuffContEnv`.

Scene configs: `conf/test_unseen_scenes_pick_from_floor_1`, `conf/test_unseen_scenes_pick_from_floor_2`,
`conf/test_unseen_items_pick_from_floor_1`, `conf/test_unseen_items_pick_from_floor_2`.

### MoveFromBoardToBoard

**Task Description:**
Approach the shelf and pick up any item with the specified name, placing it one board higher (target board).
It is assumed that there is a free space on a target board.


#### Train environments

ManiSkill environments: `MoveFromBoardToBoardVanishContEnv`, `MoveFromBoardToBoardNestleContEnv`, `MoveFromBoardToBoardDuffContEnv`.

Scene configs: `conf/move_from_board_to_board_nestle_1`, `conf/move_from_board_to_board_nestle_2`, `conf/move_from_board_to_board_vanish_1`, `conf/move_from_board_to_board_vanish_2`, `conf/move_from_board_to_board_duff_1`, `conf/move_from_board_to_board_duff_2`.

#### Test environments

ManiSkill environments with OOD target items: `MoveFromBoardToBoardFantaContEnv`, `MoveFromBoardToBoardNiveaContEnv`.

Scene configs: `conf/test_unseen_scenes_move_from_board_to_board_duff_1`, `conf/test_unseen_scenes_move_from_board_to_board_duff_2`, `conf/test_unseen_scenes_move_from_board_to_board_nestle_1`, `conf/test_unseen_scenes_move_from_board_to_board_nestle_2`, `conf/test_unseen_scenes_move_from_board_to_board_vanish_1`, `conf/test_unseen_scenes_move_from_board_to_board_vanish_2`, `conf/test_unseen_items_move_from_board_to_board_nivea_1`, `conf/test_unseen_items_move_from_board_to_board_nivea_2`, `conf/test_unseen_items_move_from_board_to_board_fanta_1`, `conf/test_unseen_items_move_from_board_to_board_fanta_2`.

## Atomic Opening and Closing Tasks

### OpenDoorShowcase

**Task Description:**
Approach the showcase and open the specified (`first`, `second`, `third`, `fourth`) door of the showcase.
The robot is spawned in close proximity to the showcase.


#### Train/Test environments

ManiSkill environments: `OpenDoorShowcaseContEnv`.

Scene configs: `conf/open_showcase`.

### CloseDoorShowcase

**Task Description:**
Approach the showcase and close the opened door of the showcase.
The robot is spawned in close proximity to the showcase.


#### Train/Test environments

ManiSkill environments: `CloseDoorShowcaseContEnv`.

Scene configs: `conf/close_showcase`.


### OpenDoorFridge

**Task Description:**
Approach the fridge and open the door.
The robot is spawned in close proximity to the fridge.


#### Train/Test environments

ManiSkill environments: `OpenDoorFridgeContEnv`.

Scene configs: `conf/open_fridge`.

### CloseDoorFridge

**Task Description:**
Approach the fridge and close the door.
The robot is spawned in close proximity to the fridge.


#### Train/Test environments

ManiSkill environments: `CloseDoorFridgeContEnv`.

Scene configs: `conf/close_fridge`.

## Composite Tasks

WIP