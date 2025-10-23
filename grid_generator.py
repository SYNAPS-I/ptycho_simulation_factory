from typing import Optional
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class GridGenerator:
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        self.positions = None
    
    def generate_positions(
        self,
        object_shape: tuple[int, int], 
        probe_lateral_shape: tuple[int, int],
    ):
        pass
    
    def plot(self, scatter_size: Optional[float] = None):
        if self.positions is None:
            raise ValueError("Positions have not been generated yet.")
        plt.scatter(self.positions[:, 1], self.positions[:, 0], s=scatter_size)
        plt.plot(self.positions[:, 1], self.positions[:, 0], linewidth=0.5, color="gray")
        plt.show()
        
    def sort_by_minimal_travel_distance(self, points: np.ndarray):
        graph = nx.complete_graph(len(points))
        for i, j in itertools.combinations(range(len(points)), 2):
            dist = math.dist(points[i], points[j])
            graph[i][j]["weight"] = dist

        # Approximate TSP route
        path = nx.approximation.traveling_salesman_problem(graph, cycle=True)
        return points[path[:-1]]
    
    def scale_grid_to_object_shape(
        self, 
        positions: np.ndarray, 
        object_shape: tuple[int, int], 
        probe_lateral_shape: tuple[int, int], 
        keep_aspect_ratio: bool = False
    ):
        height, width = [object_shape[i] - probe_lateral_shape[i] for i in range(2)]
        p_height, p_width = positions.max(0) - positions.min(0)
        if keep_aspect_ratio:
            scale = min(height / p_height, width / p_width)
            scale = [scale] * 2
        else:
            scale = [height / p_height, width / p_width]
        positions = positions * np.array(scale)
        positions = positions + np.array([probe_lateral_shape[i] // 2 for i in range(2)])
        return positions
    
    
class RasterGridGenerator(GridGenerator):
    def __init__(
        self, 
        ny: int = None, 
        nx: int = None, 
        spacing_y: float = None, 
        spacing_x: float = None, 
        *args, 
        **kwargs
    ):
        """Create a raster grid of probe positions.
        
        Parameters
        ----------
        ny : int
            Number of positions in the y direction.
        nx : int
            Number of positions in the x direction.
        spacing_y : float
            Spacing in the y direction.
        spacing_x : float
            Spacing in the x direction.
        """
        super().__init__(*args, **kwargs)
        self.ny = ny
        self.nx = nx
        self.spacing_y = spacing_y
        self.spacing_x = spacing_x
            
    def generate_positions(
        self,
        object_shape: tuple[int, int], 
        probe_lateral_shape: tuple[int, int],
    ):
        margin = [probe_lateral_shape[i] // 2 for i in range(len(probe_lateral_shape))]
        if self.ny is not None:
            assert self.spacing_y is None
            y = np.linspace(margin[0], object_shape[0] - margin[0] - 2, self.ny)
        else:
            assert self.spacing_y is not None
            y = np.arange(margin[0], object_shape[0] - margin[0] - 1, self.spacing_y)
        if self.nx is not None:
            assert self.spacing_x is None
            x = np.linspace(margin[1], object_shape[1] - margin[1] - 2, self.nx)
        else:
            assert self.spacing_x is not None
            x = np.arange(margin[1], object_shape[1] - margin[1] - 1, self.spacing_x)
        
        if len(x) == 0 or len(y) == 0:
            return []
        y, x = np.meshgrid(y, x, indexing="ij")
        positions = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)
        positions = positions - positions.mean(0)
        positions = np.flipud(positions).copy()
        self.positions = positions
        return positions


class FermatSpiralGridGenerator(GridGenerator):
    def __init__(
        self,
        keep_aspect_ratio: bool = True,
        sort_by_minimal_travel_distance: bool = False,
        *args, 
        **kwargs
    ):
        """Create a Fermat spiral grid of probe positions.
        
        Parameters
        ----------
        keep_aspect_ratio : bool
            If True, the aspect ratio of the grid is kept when scaling
            to match the object shape.
        sort_by_minimal_travel_distance : bool
            If True, the positions are sorted by the minimal travel distance
            between the positions. This may take extra time when the number of
            points is large.
        """
        super().__init__(*args, **kwargs)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.sort_by_minimal_travel_distance = sort_by_minimal_travel_distance
    
    def generate_positions(
        self,
        object_shape: tuple[int, int], 
        probe_lateral_shape: tuple[int, int],
        n_points: int = 100,
    ):
        phi_0 = np.deg2rad(137.508)
        n = np.arange(n_points)
        c = 1.0
        r = c * np.sqrt(n)
        theta = n * phi_0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions = np.stack([y, x], axis=1)
        positions = positions - positions.mean(0)
        if self.sort_by_minimal_travel_distance:
            positions = self.sort_by_minimal_travel_distance(positions)
        positions = self.scale_grid_to_object_shape(
            positions, object_shape, probe_lateral_shape, keep_aspect_ratio=self.keep_aspect_ratio
        )
        self.positions = positions
        return positions
    

if __name__ == "__main__":
    object_shape = (1024, 1024)
    probe_lateral_shape = (128, 128)
    grid_generator = RasterGridGenerator(spacing_x=10, spacing_y=10)
    grid_generator.generate_positions(object_shape, probe_lateral_shape)
    grid_generator.plot()
