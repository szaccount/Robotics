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

MAX_ANGLE = 2 * math.pi
MAX_ANGLE_FT = FT(MAX_ANGLE)


class Metric_Rod(Metric):
    """
    Implementation of a metric that takes into account the rotation of the rod.
    """

    @staticmethod
    def dist(
        pos1,
        pos2,
        clockwise,
        weight_points=1,
        weight_rotation=1,
        points_metric=Metric_Euclidean,
    ):
        """
        Return distance measure between two positions for the rod, from pos1 to pos2.
        Takes into account the rotation of the rod and if its clockwise or counter clockwise.
        Also, allows specifying weights for the distance between points and for the rotation part.

        :param pos1: first position.
        :type pos1: (:class:`~discopygal.bindings.Point_2`, :class:`~discopygal.bindings.FT`)
        :param pos2: second position
        :type pos2: (:class:`~discopygal.bindings.Point_2`, :class:`~discopygal.bindings.FT`)
        :param clockwise: direction of the angle change
        :type clockwise: bool
        :param weight_points: weight of the points distance part
        :param weight_points: :class:`~discopygal.bindings.FT`
        :param weight_rotation: weight of the rotation distance part
        :param weight_rotation: :class:`~discopygal.bindings.FT`

        :return: distance measure between pos1 and pos2
        :rtype: :class:`~discopygal.bindings.FT`
        """
        pos1_p, pos1_theta = pos1[0], pos1[1]
        pos2_p, pos2_theta = pos2[0], pos2[1]
        points_dist = points_metric.dist(pos1_p, pos2_p)
        if clockwise:
            if pos2_theta <= pos1_theta:
                rot_dist = pos1_theta - pos2_theta
            else:
                rot_dist = pos1_theta + (MAX_ANGLE_FT - pos2_theta)
        else:
            if pos2_theta <= pos1_theta:
                rot_dist = pos2_theta + (MAX_ANGLE_FT - pos1_theta)
            else:
                rot_dist = pos2_theta - pos1_theta
        w_points_FT = FT(weight_points)
        w_rotation_FT = FT(weight_rotation)
        d_squared = (w_points_FT * (points_dist * points_dist)) + (
            w_rotation_FT * (rot_dist * rot_dist)
        )
        d = math.sqrt(d_squared.to_double())
        return FT(d)


class RodRRT(Solver):
    """
    Implementation of an RRT solver, modified for rod robot.
    Supports single-robot motion planning.

    Points are tuples of (Point_2, FT) of position and angle, representing SE(2).

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param eta: max length for new RRT edge.
    :type eta: :class:`~discopygal.bindings.FT`
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
        eta=1,
        bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
        nearest_neighbors=None,  #!!!!!!!!!!!!!!!! not needed
        metric=None,
        sampler=None,
    ):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks

        self.eta = FT(eta)

        self.nearest_neighbors: NearestNeighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()

        self.metric: Metric = metric
        if self.metric is None:
            self.metric = Metric_Rod

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
            "num_landmarks": ("Number of Landmarks:", 500, int),
            "eta": ("Eta, max distance for RRT edge:", 1, FT),
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
        return RodRRT(
            d["num_landmarks"],
            FT(d["eta"]),
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
        sample = (self.sampler.sample(), FT(random.random() * MAX_ANGLE))
        robot = self.scene.robots[0]
        while not self.collision_detection[robot].is_point_valid(sample):
            sample = (self.sampler.sample(), FT(random.random() * MAX_ANGLE))
        return sample

    def point2vec3(self, point):
        """
        Convert a point (xy, theta) to a 3D vector
        """
        return Point_d(
            3,
            [point[0].x().to_double(), point[0].y().to_double(), point[1].to_double()],
        )

    def point2vec3WithDegrees(self, point):
        """
        Convert a point (xy, theta) to a 3D vector, with its angle in degrees
        """
        if type(point[0]) == Point_2:
            point_vec = self.point2vec3(point)
        else:
            point_vec = point

        return Point_d(
            3,
            [point_vec[0], point_vec[1], (point_vec[2] / (MAX_ANGLE)) * 360],
        )

    def steer_to_point(self, nearest_node, rand_node, is_clockwise):
        """
        Computes the steer point for RRT.
        """
        d = self.metric.dist(nearest_node, rand_node, is_clockwise)
        ratio = self.eta / d
        comp_ratio = FT(1) - ratio
        if ratio >= 1:
            # The distance to the new point is smaller than the max allowed
            return rand_node
        # Finding node in between which is at distance `eta` from `nearest_node`
        p_nearest, theta_nearest = nearest_node[0], nearest_node[1]
        p_rand, theta_rand = rand_node[0], rand_node[1]
        x_coor_new = ratio * (p_rand.x()) + comp_ratio * (p_nearest.x())
        y_coor_new = ratio * (p_rand.y()) + comp_ratio * (p_nearest.y())
        if is_clockwise:
            if theta_rand <= theta_nearest:
                theta_new = ratio * (theta_rand) + comp_ratio * (theta_nearest)
            else:
                theta_nearest_tag = -theta_nearest
                theta_rand_tag = MAX_ANGLE_FT - theta_rand
                theta_new = ratio * (theta_rand_tag) + comp_ratio * (theta_nearest_tag)
                if theta_new < 0:
                    theta_new = -theta_new
                else:
                    theta_new = MAX_ANGLE_FT - theta_new
        else:
            if theta_rand >= theta_nearest:
                theta_new = ratio * (theta_rand) + comp_ratio * (theta_nearest)
            else:
                theta_nearest_tag = -(MAX_ANGLE_FT - theta_nearest)
                theta_rand_tag = theta_rand
                theta_new = ratio * (theta_rand_tag) + comp_ratio * (theta_nearest_tag)
                if theta_new < 0:
                    # MAX_ANGLE_FT - (-1 * theta_new)
                    theta_new = MAX_ANGLE_FT + theta_new

        return (Point_2(x_coor_new, y_coor_new), theta_new)

    def find_nearest_node(self, p_rand):
        """
        Returns (nearest_node, is_clockwise)
        """
        nodes = list(self.roadmap.nodes)
        is_clockwise = True
        nearest_clockwise = nodes[0]
        min_d_clockwise = self.metric.dist(nearest_clockwise, p_rand, is_clockwise)
        for node in nodes[1:]:
            d = self.metric.dist(node, p_rand, is_clockwise)
            if d < min_d_clockwise:
                min_d_clockwise = d
                nearest_clockwise = node

        is_clockwise = False
        nearest_counter = nodes[0]
        min_d_counter = self.metric.dist(nearest_counter, p_rand, is_clockwise)
        for node in nodes[1:]:
            d = self.metric.dist(node, p_rand, is_clockwise)
            if d < min_d_counter:
                min_d_counter = d
                nearest_counter = node

        if min_d_clockwise < min_d_counter:
            return (nearest_clockwise, True)
        return (nearest_counter, False)

    def same_configuration(self, p1, p2):
        p1_point, p1_angle = p1[0], p1[1]
        p2_point, p2_angle = p2[0], p2[1]
        return (
            p1_point.x() == p2_point.x()
            and p1_point.y() == p2_point.y()
            and p1_angle == p2_angle
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
        # Build the RRT
        ################
        self.roadmap = nx.DiGraph()

        # Add start & end points
        self.start = scene.robots[0].start
        self.end = scene.robots[0].end
        self.roadmap.add_node(self.start)

        added_end = False
        # Add points to the tree
        for i in range(self.num_landmarks):
            p_rand = self.sample_free()
            # !!!!!!!!!!!!!!!!!!!!! maybe check not adding existing node
            # !!!!!!!!!!!!!!!!!!!!! once in 100 rounds try to add the target node
            if i % 100 == 0 and not added_end:
                # Trying to add the end node
                p_rand = self.end
            nearest_node, is_clockwise = self.find_nearest_node(p_rand)
            p_new = self.steer_to_point(nearest_node, p_rand, is_clockwise)
            # !!!!!!!! probably need to switch order
            if self.collision_free(nearest_node, p_new, is_clockwise):
                if self.same_configuration(self.end, p_new):
                    # print("Same configuration ##########", file=self.writer)
                    added_end = True
                self.roadmap.add_node(p_new)
                weight = self.metric.dist(nearest_node, p_new, is_clockwise).to_double()
                self.roadmap.add_edge(
                    nearest_node, p_new, weight=weight, clockwise=is_clockwise
                )
            if i % 100 == 0 and self.verbose:
                print("Tried adding", i, "landmarks in RRT", file=self.writer)

        # Try adding the end point
        p_new = self.end
        nearest_node, is_clockwise = self.find_nearest_node(p_new)
        # print(f"nearest node to the {self.end=} is {nearest_node=}", file=self.writer)
        self.roadmap.add_node(p_new)
        if self.collision_free(nearest_node, p_new, is_clockwise):
            # print("Edge to end is collision free", file=self.writer)
            weight = self.metric.dist(nearest_node, p_new, is_clockwise).to_double()
            self.roadmap.add_edge(
                nearest_node, p_new, weight=weight, clockwise=is_clockwise
            )
        elif self.collision_free(nearest_node, p_new, (not is_clockwise)):
            # Trying by rotating to the other direction
            # print("Edge to end is collision free on second try", file=self.writer)
            weight = self.metric.dist(
                nearest_node, p_new, (not is_clockwise)
            ).to_double()
            self.roadmap.add_edge(
                nearest_node, p_new, weight=weight, clockwise=(not is_clockwise)
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
