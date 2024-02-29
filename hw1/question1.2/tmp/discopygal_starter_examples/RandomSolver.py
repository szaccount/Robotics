import random
import networkx as nx

from discopygal.solvers import Robot, Path, PathPoint, PathCollection
from discopygal.solvers.Solver import Solver
from discopygal.geometry_utils.bounding_boxes import calc_scene_bounding_box
from discopygal.geometry_utils.collision_detection import ObjectCollisionDetection
from discopygal.bindings import *


class RandomSolver(Solver):
    def __init__(self, num_landmarks, num_connections):
        super().__init__()
        # Initializing properties of solver
        self.num_landmarks = num_landmarks
        self.num_connections = num_connections

        self._collision_detection = {}
        self._start = None
        self._end = None
        self._roadmap = None

    def _sample_random_point(self):
        # Randomize a point inside the boundaries of the scene
        x = random.uniform(self._x_min, self._x_max)
        y = random.uniform(self._y_min, self._y_max)
        return Point_2(FT(x), FT(y))

    def _create_random_point(self, robots):
        point = self._sample_random_point()

        # Set a point that for all robots it won't collide with an obstacle
        is_valid_point = False
        while not is_valid_point:
            point = self._sample_random_point()
            is_valid_point = all(
                [
                    self._collision_detection[robot].is_point_valid(point)
                    for robot in robots
                ]
            )

        return point

    def _create_random_roadmap(self, robots):
        roadmap = nx.Graph()
        # Add random points
        for _ in range(self.num_landmarks):
            point = self._create_random_point(robots)
            roadmap.add_node(point)

        # Add random connections
        for _ in range(self.num_connections):
            v, u = random.sample(list(roadmap.nodes), 2)
            print(f"points {v=} {u=}")
            roadmap.add_edge(v, u, weight=1)

        for robot in robots:
            # Add starting point of robot to the graph
            roadmap.add_node(robot.start)

            # Connect start to a random point
            roadmap.add_edge(
                robot.start, *random.sample(list(roadmap.nodes), 1), weight=1
            )

            # Add ending point of robot to the graph
            roadmap.add_node(robot.end)

            # Connect to end to a random point
            roadmap.add_edge(
                robot.end, *random.sample(list(roadmap.nodes), 1), weight=1
            )

        return roadmap

    def load_scene(self, scene):
        super().load_scene(scene)
        self._x_min, self._x_max, self._y_min, self._y_max = calc_scene_bounding_box(
            self.scene
        )

        # Build collision detection for each robot
        for robot in self.scene.robots:
            self._collision_detection[robot] = ObjectCollisionDetection(
                scene.obstacles, robot
            )

    def get_graph(self):
        return self._roadmap

    def solve(self):
        self.log("Solving...")
        self._roadmap = self._create_random_roadmap(self.scene.robots)

        path_collection = (
            PathCollection()
        )  # Initialize PathCollection (stores the path for each robot)
        for i, robot in enumerate(self.scene.robots):
            self.log(f"Robot {i}")

            # Check if there is a possible path for the robot in the graph
            if not nx.algorithms.has_path(self._roadmap, robot.start, robot.end):
                self.log(f"No path found for robot {i}")
                return PathCollection()

            # Get the shortest path for the robot
            found_path = nx.algorithms.shortest_path(
                self._roadmap, robot.start, robot.end
            )
            points = [
                PathPoint(point) for point in found_path
            ]  # Convert all points to PathPoints (to make a path out of them)
            path = Path(points)  # Make a path from all PathPoints
            path_collection.add_robot_path(
                robot, path
            )  # Add the current path for the current robot to the path collection

        self.log("Successfully found a path for all robots")
        return path_collection

    @staticmethod
    def get_arguments():
        # Returns the configurable properties of the solver (presented in gui)
        # in format of: 'property_name': (Description, Default value, Type)
        return {
            "num_landmarks": ("Number of points", 5, int),
            "num_connections": ("Number of connections", 5, int),
        }
