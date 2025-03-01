import numpy as np
from scene_synthesizer.utils import PositionIterator2D
from dsynth.assets.ss_assets import WIDTH, DEPTH
from shapely.geometry import Point

class PositionIteratorPI(PositionIterator2D):
    def __init__(
        self,
        step_x,
        step_y,
        noise_std_x=0.0,
        noise_std_y=0.0,
        direction="x",
        stop_on_new_line=False,
        seed=None,
        shelf_width=WIDTH,
        shelf_depth=DEPTH
    ):
        super().__init__(seed=seed)
        self.step = np.array([step_x, step_y])
        self.noise_std_x = noise_std_x
        self.noise_std_y = noise_std_y
        self.direction = direction

        self.new_line = False
        self.stop_on_new_line = stop_on_new_line

        # if self.direction
        #     raise ValueError(f"Unknown direction: {self.direction}")
        self.start_point = None
        self.end_point = None
        self.i = 0
        self.j = 0
        self.lst_of_pos = [(1.45 * shelf_width / 4, shelf_depth / 3),
                        (1.45 * shelf_width / 4, 2 * shelf_depth / 3),
                        (shelf_width / 2, 2 * shelf_depth / 3),
                        (3 * shelf_width / 4 - 0.45 * shelf_width / 4, 2 * shelf_depth / 3),
                        (3 * shelf_width / 4 - 0.45 * shelf_width / 4, shelf_depth / 3)]
        self.counter = 0

    def __next__(self):
        while True:
            if self.stop_on_new_line and self.new_line:
                self.new_line = False
                raise StopIteration
            current_point = self.lst_of_pos[self.counter]
            self.counter += 1
            p = Point(current_point)

            if np.all(current_point > self.end_point):
                break

            if p.within(self.polygon):
                return np.array([p.x, p.y])

        raise StopIteration

    def __call__(self, support):
        if support.polygon != self.polygon:
            self.polygon = support.polygon

            minx, miny, maxx, maxy = self.polygon.bounds

            self.start_point = np.array([minx, miny])
            self.end_point = np.array([maxx, maxy])
            self.i = 0
            self.j = 0

            self.new_line = False

        return self
