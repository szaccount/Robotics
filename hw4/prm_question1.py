import sys

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


class PRM_Vertical(Solver):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    The basic implementation of a Probabilistic Road Map (PRM) solver.
    Supports multi-robot motion planning, though might be inefficient for more than
    two-three robots.

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
        # bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
        bounding_margin_width_factor=0,
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
        self.clearance_epsilon = 0.01  # !!!!!!!!!!!!!!! maybe smaller

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
                # Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
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
        return PRM(
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

    def collision_free(self, p, q):
        """
        Get two points in the configuration space and decide if they can be connected
        """

        edge = Segment_2(p, q)
        if not self.robot_collision_detection.is_edge_valid(edge):
            return False

        return True

    def sample_free(self):
        """
        Sample a free random point
        """
        sample = self.sampler.sample()
        while not self.robot_collision_detection.is_point_valid(sample):
            sample = self.sampler.sample()
        return sample

    def _binary_search_up_clear_area(self, anchor, bottom_y, top_y):
        """
        Returns the clearance from anchor to the obsacles above it.
        """
        assert top_y - bottom_y >= 0  # !!!!!!!!! delete
        if (top_y - bottom_y) < self.clearance_epsilon:
            return abs(bottom_y - anchor.y().to_double())

        mid_y = (bottom_y + top_y) / 2
        mid_p = Point_2(anchor.x(), FT(mid_y))
        segment = Segment_2(anchor, mid_p)
        if self.robot_collision_detection.is_edge_valid(segment):
            return self._binary_search_up_clear_area(anchor, mid_y, top_y)
        else:
            return self._binary_search_up_clear_area(anchor, bottom_y, mid_y)

    def _binary_search_down_clear_area(self, anchor, bottom_y, top_y):
        """
        Returns the clearance from anchor to the obsacles below it.
        """
        assert top_y - bottom_y >= 0  # !!!!!!!!! delete
        if (top_y - bottom_y) < self.clearance_epsilon:
            return abs(anchor.y().to_double() - top_y)

        mid_y = (bottom_y + top_y) / 2
        mid_p = Point_2(anchor.x(), FT(mid_y))
        segment = Segment_2(anchor, mid_p)
        if self.robot_collision_detection.is_edge_valid(segment):
            return self._binary_search_down_clear_area(anchor, bottom_y, mid_y)
        else:
            return self._binary_search_down_clear_area(anchor, mid_y, top_y)

    def vertical_clearance(self, point):
        """
        Calculates the vertical clearance of a point.
        """
        min_x, max_x, min_y, max_y = self._bounding_box
        y_range_bottom = min_y.to_double()
        y_range_top = max_y.to_double()
        p_y = point.y().to_double()
        up_clearance = self._binary_search_up_clear_area(point, p_y, y_range_top)
        down_clearance = self._binary_search_down_clear_area(point, y_range_bottom, p_y)
        clearance = min(up_clearance, down_clearance)
        return clearance

    def point_weight_by_clearance(self, p):
        """
        Returns the weight of p based on its clearance
        """
        p_clearance = self.vertical_clearance(p)
        if p_clearance <= 0:
            p_weight = sys.float_info.max
        else:
            p_weight = 1 / p_clearance

        return p_weight

    # def edge_weight_by_clearance(self, p, q):
    #     """
    #     Returns the weight of the edge between p,q based on their clearance
    #     """
    #     p_weight = self.point_weight_by_clearance(p)
    #     q_weight = self.point_weight_by_clearance(q)
    #     weight = max(p_weight, q_weight)
    #     return weight

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection for the robot
        robot = scene.robots[0]
        self.robot_collision_detection = collision_detection.ObjectCollisionDetection(
            scene.obstacles, robot
        )

        ################
        # Build the PRM
        ################
        self.roadmap = nx.Graph()

        # Add start & end points
        self.start = robot.start
        self.end = robot.end
        self.roadmap.add_node(self.start)
        self.roadmap.add_node(self.end)

        weight_dict = dict()
        weight_dict[self.start] = self.point_weight_by_clearance(self.start)
        weight_dict[self.end] = self.point_weight_by_clearance(self.end)

        # Add valid points
        for i in range(self.num_landmarks):
            p_rand = self.sample_free()
            self.roadmap.add_node(p_rand)
            weight_dict[p_rand] = self.point_weight_by_clearance(p_rand)
            if i % 100 == 0 and self.verbose:
                print("added", i, "landmarks in PRM", file=self.writer)

        self.nearest_neighbors.fit(list(self.roadmap.nodes))

        # Connect all points to their k nearest neighbors
        for cnt, point in enumerate(self.roadmap.nodes):
            neighbors = self.nearest_neighbors.k_nearest(point, self.k + 1)
            for neighbor in neighbors:
                if self.collision_free(neighbor, point):
                    if point not in weight_dict:
                        weight_dict[point] = self.point_weight_by_clearance(point)
                    if neighbor not in weight_dict:
                        weight_dict[neighbor] = self.point_weight_by_clearance(neighbor)
                    edge_weight = max(weight_dict[point], weight_dict[neighbor])
                    self.roadmap.add_edge(
                        point,
                        neighbor,
                        # !!!!!!!!!!!!!!!!!!!! need to change the weight for clearance
                        # weight=self.metric.dist(point, neighbor).to_double(),
                        # weight=self.edge_weight_by_clearance(point, neighbor),
                        weight=edge_weight,
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
        for i, robot in enumerate(self.scene.robots):
            points = []
            for point in tensor_path:
                points.append(PathPoint(Point_2(point[2 * i], point[2 * i + 1])))
            path = Path(points)
            path_collection.add_robot_path(robot, path)

        if self.verbose:
            print("successfully found a path...", file=self.writer)

        return path_collection
