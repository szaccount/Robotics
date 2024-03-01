import math
import random

import networkx as nx
import matplotlib.pyplot as plt

from discopygal.solvers import Robot, RobotDisc, RobotPolygon, RobotRod
from discopygal.solvers import Obstacle, ObstacleDisc, ObstaclePolygon, Scene
from discopygal.solvers import PathPoint, Path, PathCollection

from discopygal.solvers.samplers import Sampler, Sampler_Uniform
from discopygal.solvers.metrics import Metric, Metric_Euclidean
from discopygal.solvers.nearest_neighbors import (
    NearestNeighbors,
    NearestNeighbors_sklearn,
)
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.Solver import Solver


class ImprovedRodPRM(Solver):
    """
    The basic implementation of a Probabilistic Road Map (PRM) solver, modified for rod robot.
    Supports single-robot motion planning.

    Points are tuples of (Point_2, FT) of position and angle, representing SE(2).

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param k: number of nearest neighbors to connect
    :type k: :class:`int`
    :param nearest_neighbors: a nearest neighbors algorithm. if None then use sklearn implementation
    :type nearest_neighbors: :class:`~discopygal.solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param metric: a metric for weighing edges, can be different then the nearest_neighbors metric!
        If None then use euclidean metric
    :type metric: :class:`~discopygal.solvers.metrics.Metric` or :class:`None`
    :param sampler: sampling algorithm/method. if None then use uniform sampling
    :type sampler: :class:`~discopygal.solvers.samplers.Sampler`
    """

    def __init__(
        self,
        num_landmarks,
        k,
        bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
        nearest_neighbors=None,
        metric=None,
        sampler=None,
    ):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.k = k

        self.nearest_neighbors: NearestNeighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()

        self.metric: Metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

        self.sampler: Sampler = sampler
        if self.sampler is None:
            self.sampler = Sampler_Uniform()

        self.roadmap = None
        self.collision_detection = {}
        self.start = None
        self.end = None

    @staticmethod
    def get_arguments():
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.
        Should be overridded by solvers.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        return {
            "num_landmarks": ("Number of Landmarks:", 1000, int),
            "k": ("K for nearest neighbors:", 15, int),
            "bounding_margin_width_factor": (
                "Margin width factor (for bounding box):",
                0,
                FT,
            ),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return ImprovedRodPRM(
            d["num_landmarks"],
            d["k"],
            FT(d["bounding_margin_width_factor"]),
            None,
            None,
            None,
        )

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridded by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        return self.roadmap

    def collision_free(self, p, q, clockwise):
        """
        Get two points in the configuration space and decide if they can be connected
        """
        # Check validity of each edge seperately
        for robot in self.scene.robots:
            xy_edge = Segment_2(p[0], q[0])
            if not self.collision_detection[robot].is_edge_valid(
                (xy_edge, FT(p[1]), FT(q[1])), clockwise
            ):
                return False
            break  # Only one robot is supported
        return True

    def sample_free(self):
        """
        Sample a free random point
        """
        sample = (self.sampler.sample(), FT(random.random() * 2 * math.pi))
        robot = self.scene.robots[0]
        while not self.collision_detection[robot].is_point_valid(sample):
            sample = (self.sampler.sample(), FT(random.random() * 2 * math.pi))
        return sample

    def point2vec3(self, point):
        """
        Convert a point (xy, theta) to a 3D vector
        """
        return Point_d(
            3,
            [point[0].x().to_double(), point[0].y().to_double(), point[1].to_double()],
        )

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection for each robot
        for robot in scene.robots:
            self.collision_detection[
                robot
            ] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

        ################
        # Build the PRM
        ################
        self.roadmap = nx.DiGraph()

        # Add start & end points
        self.start = scene.robots[0].start
        self.end = scene.robots[0].end
        self.roadmap.add_node(self.start)
        self.roadmap.add_node(self.end)

        # Add valid points
        for i in range(self.num_landmarks):
            p_rand = self.sample_free()
            self.roadmap.add_node(p_rand)
            if i % 100 == 0 and self.verbose:
                print("added", i, "landmarks in PRM", file=self.writer)

        self.nearest_neighbors.fit(list(map(self.point2vec3, self.roadmap.nodes)))

        # Connect all points to their k nearest neighbors
        last_nodes = list(self.roadmap.nodes)
        for cnt, point in enumerate(last_nodes):
            neighbors = self.nearest_neighbors.k_nearest(
                self.point2vec3(point), self.k + 1
            )
            for neighbor in neighbors:
                neighbor = (Point_2(neighbor[0], neighbor[1]), neighbor[2])
                for clockwise in [True, False]:
                    if self.collision_free(point, neighbor, clockwise):
                        weight = self.metric.dist(point[0], neighbor[0]).to_double()
                        self.roadmap.add_edge(
                            point, neighbor, weight=weight, clockwise=clockwise
                        )
                        self.roadmap.add_edge(
                            neighbor, point, weight=weight, clockwise=(not clockwise)
                        )

            if cnt % 100 == 0 and self.verbose:
                print(
                    "connected",
                    cnt,
                    "landmarks to their nearest neighbors",
                    file=self.writer,
                )

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        if not nx.algorithms.has_path(self.roadmap, self.start, self.end):
            if self.verbose:
                print("no path found...", file=self.writer)
            return PathCollection()

        # Convert from a sequence of Point_d points to PathCollection
        tensor_path = nx.algorithms.shortest_path(
            self.roadmap, self.start, self.end, weight="weight"
        )
        path_collection = PathCollection()
        for _, robot in enumerate(self.scene.robots):
            points = []
            for i, point in enumerate(tensor_path):
                if point != self.end:
                    clockwise = self.roadmap.get_edge_data(
                        tensor_path[i], tensor_path[i + 1]
                    )["clockwise"]
                points.append(
                    PathPoint(
                        point[0], data={"theta": point[1], "clockwise": clockwise}
                    )
                )
            path = Path(points)
            path_collection.add_robot_path(robot, path)

        if self.verbose:
            print("successfully found a path...", file=self.writer)

        return path_collection
