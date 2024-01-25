import random
import networkx as nx

from discopygal.solvers import Robot, Path, PathPoint, PathCollection
from discopygal.solvers.Solver import Solver
from discopygal.geometry_utils.bounding_boxes import calc_scene_bounding_box
from discopygal.geometry_utils.collision_detection import ObjectCollisionDetection
from discopygal.bindings import *

from sklearn.neighbors import NearestNeighbors


class RoombaSolver(Solver):
    def __init__(self, sample_points, num_neighbors):
        """
        Param sample_points: represents general number of uniform samples on the map.
        Param num_neighbors: max number of neighbors trying to connect each point in the roadmap.
        """
        super().__init__()
        # Initializing properties of solver
        self.sample_points = sample_points
        self.num_neighbors = num_neighbors

        self._collision_detection = {}
        self._start = None
        self._end = None
        self._roadmap = None

    def _sample_random_point(self):
        # Randomize a point inside the boundaries of the scene
        x = random.uniform(self._x_min, self._x_max)
        y = random.uniform(self._y_min, self._y_max)
        return Point_2(FT(x), FT(y))

    def _connect_node_to_nodes_if_possible(
        self, new_node, nodes, indxs_to_connect, robot, roadmap
    ):
        """
        Connects the new node to the nodes in the passed indexes if possible.
        """
        for neigh_index in indxs_to_connect:
            neigh_node = nodes[neigh_index]
            # Check if can connect the two nodes
            can_connect = self._collision_detection[robot].is_edge_valid(
                Segment_2(new_node, neigh_node)
            )
            if can_connect:
                roadmap.add_edge(new_node, neigh_node, weight=1)

    def _create_random_roadmap(self, robot):
        roadmap = nx.Graph()
        # Add random points
        for _ in range(self.sample_points):
            point = self._sample_random_point()
            is_free_point = self._collision_detection[robot].is_point_valid(point)
            if is_free_point:
                roadmap.add_node(point)

        # Adding start and end of robot as nodes
        roadmap.add_node(robot.start)
        roadmap.add_node(robot.end)
        nodes_list = list(roadmap.nodes)
        points_list = []
        for node in nodes_list:
            x = node.x().to_double()
            y = node.y().to_double()
            points_list.append([x, y])

        # plus 1 as the point itself will return as nearest neighbor
        neigh = NearestNeighbors(n_neighbors=self.num_neighbors + 1, algorithm="brute")
        neigh.fit(points_list)

        # Finding neighbors and connecting to them
        for curr_index in range(len(nodes_list)):
            curr_point = points_list[curr_index]
            # can do with distance, then gives pair of two lists where second as here and first is of distances
            indx_neighbors_of_point = neigh.kneighbors(
                [curr_point], return_distance=False
            )
            indx_neighbors_of_point = indx_neighbors_of_point.tolist()[0]
            indx_neighbors_of_point.remove(curr_index)
            curr_node = nodes_list[curr_index]
            self._connect_node_to_nodes_if_possible(
                curr_node, nodes_list, indx_neighbors_of_point, robot, roadmap
            )

        return roadmap

    def load_scene(self, scene):
        super().load_scene(scene)
        self._x_min, self._x_max, self._y_min, self._y_max = calc_scene_bounding_box(
            self.scene
        )

        robot = self.scene.robots[0]
        self._collision_detection[robot] = ObjectCollisionDetection(
            scene.obstacles, robot
        )

    def get_graph(self):
        return self._roadmap

    def solve(self):
        self.log("Solving...")
        robot = self.scene.robots[0]
        self._roadmap = self._create_random_roadmap(robot)

        path_collection = (
            PathCollection()
        )  # Initialize PathCollection (stores the path for robot)

        # Check if there is a possible path for the robot in the graph
        if not nx.algorithms.has_path(self._roadmap, robot.start, robot.end):
            self.log(f"No path found for robot")
            return PathCollection()
        # Get the shortest path for the robot
        found_path = nx.algorithms.shortest_path(self._roadmap, robot.start, robot.end)
        points = [
            PathPoint(point) for point in found_path
        ]  # Convert all points to PathPoints (to make a path out of them)
        path = Path(points)  # Make a path from all PathPoints
        path_collection.add_robot_path(
            robot, path
        )  # Add the current path for the robot to the path collection

        self.log("Successfully found a path for the robot")
        return path_collection

    @staticmethod
    def get_arguments():
        # Returns the configurable properties of the solver (presented in gui)
        # in format of: 'property_name': (Description, Default value, Type)
        return {
            "sample_points": ("Number of points to sample on map", 2000, int),
            "num_neighbors": (
                "Maximum number of neighbors of point in the roadmap",
                15,
                int,
            ),
        }
