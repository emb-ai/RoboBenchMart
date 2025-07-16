from itertools import cycle
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import random

from dsynth.scene_gen.utils import RectFixture, check_collisions

class TensorField:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.base_fields = []

        self.field = None
        self.eigen_vectors = None

    def add_grid_basis(self, direction = np.array([1., 1.]), origin=np.array([0, 0])):
        l = np.linalg.norm(direction)
        angle = np.arctan(direction[1] / (direction[0] + 1e-4))
        tensor = l * np.array([
            [np.cos(2 * angle), np.sin(2 * angle)],
            [np.sin(2 * angle), -np.cos(2 * angle)]
        ], dtype=np.float32)
        self.base_fields.append(
            {
                "type": "grid",
                "origin": origin,
                "tensor": lambda p: tensor
            }
        )

    def add_radial_basis(self, origin=np.array([0, 0])):
        origin = origin.astype(np.float32)
        self.base_fields.append(
            {
                "type": "radial",
                "origin": origin.astype(np.float32),
                "tensor": lambda p: np.array([
                    [(p[1] - origin[1]) ** 2 - (p[0] - origin[0]) ** 2, 
                     -2 * (p[0] - origin[0]) * (p[1] - origin[1])],
                    [-2 * (p[0] - origin[0]) * (p[1] - origin[1]), 
                     (origin[0] - p[0]) ** 2 - (origin[1] - p[1]) ** 2]
                ], dtype=np.float32)
            }
        )
    def add_line(self, line, closed = False, sample_step=None):
        n_points = len(line)
        line = np.array(line)
        for i in range(n_points):
            cur_point = line[i]
            if i == n_points - 1:
                if closed:
                    next_point = line[0]
                else:
                    break
            else:
                next_point = line[i + 1]

            vec = next_point - cur_point
            self.add_grid_basis(vec, cur_point)
            
            if sample_step is not None:
                while np.linalg.norm(vec) > sample_step:
                    cur_point = cur_point + vec / np.linalg.norm(vec) * sample_step
                    vec = next_point - cur_point
                    self.add_grid_basis(vec, cur_point)
    
    def add_fixture_list(self, fixture_list):
        for shelf in fixture_list:
            _, polygon = shelf.get_polygon()
            self.add_line(polygon, closed=True, sample_step=0.5)
    
    def add_boundary(self):
        boundary = np.array([[0, 0], [0, self.M], [self.N, self.M], [self.N, 0],])
        self.add_line(boundary, closed=True, sample_step=1.)
        

    def calculate_field(self, recalculate = False, decay=0.1):
        if recalculate or self.field is None or self.eigen_vectors is None:
            assert len(self.base_fields) > 0, "No basis fields"
            zero_tensor = np.array([[0., 0.], [0., 0.]], dtype=np.float32)
            # TODO: redo
            M = int(self.M)
            N = int(self.N)
            self.field = [[zero_tensor for _ in range(M)] for _ in range(N)]
            self.eigen_vectors = [[zero_tensor for _ in range(M)] for _ in range(N)]
            
            for i in range(N):
                for j in range(M):
                    cur_point = np.array([i, j], dtype=np.float32)
                    for basis_field_meta in self.base_fields:
                        weight = np.exp(-decay * np.linalg.norm(
                               cur_point - basis_field_meta["origin"]
                            ))
                        # weight = 1.
                        self.field[i][j] = self.field[i][j] + weight * basis_field_meta["tensor"](cur_point)

            self.field = np.array(self.field)

            for i in range(N):
                for j in range(M):
                    eigenvalues, eigenvectors = np.linalg.eig(self.field[i, j])
                    idxs = np.argsort(eigenvalues)[::-1]
                    self.eigen_vectors[i][j] = eigenvectors.T[idxs]

            self.eigen_vectors = np.array(self.eigen_vectors)

    def vis_field(self):
        M = int(self.M)
        N = int(self.N)
        X, Y = np.meshgrid(np.arange(N), np.arange(M))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        U = self.eigen_vectors[:, :, 0, 0]
        V = self.eigen_vectors[:, :, 0, 1]
        U_minor = self.eigen_vectors[:, :, 1, 0]
        V_minor = self.eigen_vectors[:, :, 1, 1]
        # ax[0].axis('equal')
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xlim(0, N-1)
        ax[0].set_ylim(0, M-1)
        ax[0].quiver(X.T, Y.T, U, V, color='r')
        ax[0].quiver(X.T, Y.T, U_minor, V_minor, color='g')
        ax[0].quiver(X.T, Y.T, -U, -V, color='r')
        ax[0].quiver(X.T, Y.T, -U_minor, -V_minor, color='g')

        # ax[1].axis('equal')
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_xlim(0, N-1)
        ax[1].set_ylim(0, M-1)
        # ax[1].axis('equal')
        ax[1].streamplot(X, Y, U.T, V.T, density=2, color='r')#, start_points=seed_points.T)
        # ax[1].streamplot(X, Y, U_minor.T, V_minor.T, density=2, color='g')#, start_points=seed_points.T)
        return fig, ax 
        # plt.show()


def is_horizontal(vec, thresh=0.2):
    unit_vec = vec / np.linalg.norm(vec)
    if 1 - np.abs(np.dot(unit_vec, [1, 0])) < thresh:
        return True
    return False

def is_vertical(vec, thresh=0.2):
    unit_vec = vec / np.linalg.norm(vec)
    if 1 - np.abs(np.dot(unit_vec, [0, 1])) < thresh:
        return True
    return False

def place_shelves(tf: TensorField, 
                  sample_rects: RectFixture,
                  rng: random.Random,
                  start_point=np.array([1., 1.]), 
                  passage_width=0.5, 
                  skip_shelf_prob=0., 
                  thresh=0.2,
                 scene_fixtures = []):
    shelves = []
    cur_position = start_point.copy()

    sample_rects_sampler = cycle(sample_rects)
    rect_example = next(sample_rects_sampler)

    shelf_l, shelf_w = rect_example.l, rect_example.w
    occupancy_width = rect_example.occupancy_width

    X, Y = np.meshgrid(np.arange(tf.N), np.arange(tf.M))
    points = np.vstack((X.T.reshape((-1)), Y.T.reshape((-1)))).T
    U = tf.eigen_vectors[:, :, 0, 0]
    V = tf.eigen_vectors[:, :, 0, 1]
    
    def _get_major_eigen(p):
        u_interp = interpolate.griddata(points, U.reshape((-1)), ([p[0]], [p[1]]), method='cubic')
        v_interp = interpolate.griddata(points, V.reshape((-1)), ([p[0]], [p[1]]), method='cubic')
        return np.array([u_interp[0], v_interp[0]])
    

    x_step = shelf_l + 1e-2
    y_step = shelf_w + passage_width 

    is_first_shelf = True
    while True:
        eigen_major = _get_major_eigen(cur_position)

        
        if is_horizontal(eigen_major, thresh):
            if rng.random() > skip_shelf_prob:
                shelf = RectFixture(x=cur_position[0], y=cur_position[1], w=shelf_w, l=shelf_l, 
                                    occupancy_width=occupancy_width,
                                    name=rect_example.name, asset_name=rect_example.asset_name)
                if shelf.is_valid(tf.N - 1, tf.M - 1) and not check_collisions(shelf, shelves) and not check_collisions(shelf, scene_fixtures):
                    shelves.append(shelf)
                    is_first_shelf = False
        
        x_step = 0.1 if is_first_shelf else shelf_l + 1e-2
        
        if cur_position[0] + x_step < tf.N:
            cur_position[0] += x_step
        else:
            if cur_position[1] + y_step < tf.M:
                cur_position[0] = start_point[0]
                cur_position[1] += y_step

                if not is_first_shelf:
                    rect_example = next(sample_rects_sampler)
                    shelf_l, shelf_w = rect_example.l, rect_example.w
                    occupancy_width = rect_example.occupancy_width

            else:
                break

    cur_position = start_point.copy()

    sample_rects_sampler = cycle(sample_rects)
    rect_example = next(sample_rects_sampler)

    shelf_l, shelf_w = rect_example.l, rect_example.w
    occupancy_width = rect_example.occupancy_width

    x_step = shelf_w + passage_width 
    y_step = shelf_l + 1e-2 


    is_first_shelf = True
    while True:
        eigen_major = _get_major_eigen(cur_position)
        
        if is_vertical(eigen_major, thresh):
            if rng.random() > skip_shelf_prob:
                shelf = RectFixture(x=cur_position[0], y=cur_position[1], w=shelf_w, l=shelf_l, 
                                    orientation = 'vertical', occupancy_width=occupancy_width,
                                    name=rect_example.name, asset_name=rect_example.asset_name)
                if shelf.is_valid(tf.N - 1, tf.M - 1) and not check_collisions(shelf, shelves) and not check_collisions(shelf, scene_fixtures):
                    shelves.append(shelf)
                    is_first_shelf = False
                    
        y_step = 0.1 if is_first_shelf else shelf_l + 1e-2 
        
        if cur_position[1] + y_step < tf.M:
            cur_position[1] += y_step
        else:
            if cur_position[0] + x_step < tf.N:
                cur_position[1] = start_point[1]
                cur_position[0] += x_step

                if not is_first_shelf:
                    rect_example = next(sample_rects_sampler)
                    shelf_l, shelf_w = rect_example.l, rect_example.w
                    occupancy_width = rect_example.occupancy_width

            else:
                break
    
    return shelves
    

