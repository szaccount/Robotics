import random
import networkx as nx

from discopygal.solvers import Robot, Path, PathPoint, PathCollection
from discopygal.solvers.Solver import Solver
from discopygal.geometry_utils.bounding_boxes import calc_scene_bounding_box
from discopygal.geometry_utils.collision_detection import ObjectCollisionDetection
from discopygal.bindings import *

# !!!!!! my imports
import numpy as np
from sklearn.neighbors import NearestNeighbors


class RoombaSolver(Solver):
    def __init__(self, sample_points, samples_for_edge, num_neighbors):
        """
        Param sample_points: represents general number of uniform samples on the map.
        Param samples_for_edge: number of samples we take to verify edge is valid to move on.
        Param num_neighbors: max number of neighbors trying to connect each point in the roadmap.
        """
        super().__init__()
        # Initializing properties of solver
        self.sample_points = sample_points
        self.samples_for_edge = samples_for_edge
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

    # Was originaly, DELETE
    # def _create_random_point(self, robots):
    #     point = self._sample_random_point()

    #     # Set a point that for all robots it won't collide with an obstacle
    #     is_valid_point = False
    #     while not is_valid_point:
    #         point = self._sample_random_point()
    #         is_valid_point = all(
    #             [
    #                 self._collision_detection[robot].is_point_valid(point)
    #                 for robot in robots
    #             ]
    #         )

    #     return point

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

        for curr_index in range(len(nodes_list)):
            curr_point = points_list[curr_index]
            # can do with distance, then gives pair of two lists where second as here and first is of distances
            indx_neighbors_of_point = neigh.kneighbors(
                [curr_point], return_distance=False
            )
            # DELETE !!!!!!!!!!!
            # print(
            #     f"for point {curr_index=} the neighbors indexes are {indx_neighbors_of_point=}"
            # )
            indx_neighbors_of_point = indx_neighbors_of_point.tolist()[0]
            indx_neighbors_of_point.remove(curr_index)
            curr_node = nodes_list[curr_index]
            # !!!! extracted to function
            # for neigh_index in indx_neighbors_of_point:
            #     neigh_node = nodes_list[neigh_index]
            #     # Check if can connect the two nodes
            #     can_connect = self._collision_detection[robot].is_edge_valid(
            #         Segment_2(curr_node, neigh_node)
            #     )
            #     if can_connect:
            #         roadmap.add_edge(curr_node, neigh_node, weight=1)
            self._connect_node_to_nodes_if_possible(
                curr_node, nodes_list, indx_neighbors_of_point, robot, roadmap
            )

        # # Add random connections
        # for _ in range(10):
        #     v, u = random.sample(list(roadmap.nodes), 2)
        #     print(f"points {u=} {v=}")
        #     roadmap.add_edge(v, u, weight=1)
        #
        # for robot in robots:
        #     # Add starting point of robot to the graph
        #     roadmap.add_node(robot.start)
        #     # Connect start to a random point
        #     roadmap.add_edge(
        #         robot.start, *random.sample(list(roadmap.nodes), 1), weight=1
        #     )
        #     # Add ending point of robot to the graph
        #     roadmap.add_node(robot.end)
        #     # Connect to end to a random point
        #     roadmap.add_edge(
        #         robot.end, *random.sample(list(roadmap.nodes), 1), weight=1
        #     )

        return roadmap

    def load_scene(self, scene):
        super().load_scene(scene)
        self._x_min, self._x_max, self._y_min, self._y_max = calc_scene_bounding_box(
            self.scene
        )

        # Build collision detection for each robot
        # !!!!!! DELETE
        assert len(self.scene.robots) == 1, "Should only be single robot"
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
        )  # Initialize PathCollection (stores the path for each robot)
        # for i, robot in enumerate(self.scene.robots):
        #     self.log(f"Robot {i}")

        #     # Check if there is a possible path for the robot in the graph
        #     if not nx.algorithms.has_path(self._roadmap, robot.start, robot.end):
        #         self.log(f"No path found for robot {i}")
        #         return PathCollection()

        #     # Get the shortest path for the robot
        #     found_path = nx.algorithms.shortest_path(
        #         self._roadmap, robot.start, robot.end
        #     )
        #     points = [
        #         PathPoint(point) for point in found_path
        #     ]  # Convert all points to PathPoints (to make a path out of them)
        #     path = Path(points)  # Make a path from all PathPoints
        #     path_collection.add_robot_path(
        #         robot, path
        #     )  # Add the current path for the current robot to the path collection

        # Check if there is a possible path for the robot in the graph
        print("HERE !!!!")
        if not nx.algorithms.has_path(self._roadmap, robot.start, robot.end):
            self.log(f"No path found for robot")
            print("no path !!!!")
            return PathCollection()
        # Get the shortest path for the robot
        print("After if !!!!")
        found_path = nx.algorithms.shortest_path(self._roadmap, robot.start, robot.end)
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
            "sample_points": ("Number of points to sample on map", 2000, int),
            "samples_for_edge": (
                "Number of sample along desired edge to verify it is valid",
                50,
                int,
            ),
            "num_neighbors": (
                "Maximum number of neighbors of point in the roadmap",
                15,
                int,
            ),
        }
