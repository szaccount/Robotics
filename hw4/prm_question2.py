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


class PRM_TwoRobots(Solver):
    """
    Improved PRM solver for two robots.

    :param num_landmarks: number of landmarks to sample (per HGraph iteration)
    :type num_landmarks: :class:`int`
    :param k: number of nearest neighbors to connect
    :type k: :class:`int`
    :param num_iterations: number of separate PRM trials
    :type num_iterations: :class:`int`
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
        num_iterations,
        # bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
        bounding_margin_width_factor=0,
        nearest_neighbors=None,
        metric=None,
        sampler=None,
    ):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.k = k
        self.num_iterations = num_iterations

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

        self.paths_total_length = None

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
            "num_iterations": ("Num separate PRM iterations:", 3, int),
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
            d["num_iterations"],
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
        p_list = conversions.Point_d_to_Point_2_list(p)
        q_list = conversions.Point_d_to_Point_2_list(q)

        # Check validity of each edge seperately
        for i, robot in enumerate(self.scene.robots):
            edge = Segment_2(p_list[i], q_list[i])
            if not self.collision_detection[robot].is_edge_valid(edge):
                return False

        # Check validity of coordinated robot motion
        for i, robot1 in enumerate(self.scene.robots):
            for j, robot2 in enumerate(self.scene.robots):
                if j <= i:
                    continue
                edge1 = Segment_2(p_list[i], q_list[i])
                edge2 = Segment_2(p_list[j], q_list[j])
                if collision_detection.collide_two_robots(robot1, edge1, robot2, edge2):
                    return False

        return True

    def sample_free(self):
        """
        Sample a free random point
        """
        p_rand = []
        for robot in self.scene.robots:
            sample = self.sampler.sample()
            while not self.collision_detection[robot].is_point_valid(sample):
                sample = self.sampler.sample()
            p_rand.append(sample)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)
        return p_rand

    def _create_prm(self, roadmap):
        # Add valid points
        for i in range(self.num_landmarks):
            p_rand = self.sample_free()
            roadmap.add_node(p_rand)
            if i % 100 == 0 and self.verbose:
                print("added", i, "landmarks in PRM", file=self.writer)

        self.nearest_neighbors.fit(list(roadmap.nodes))

        # Connect all points to their k nearest neighbors
        for cnt, point in enumerate(roadmap.nodes):
            neighbors = self.nearest_neighbors.k_nearest(point, self.k + 1)
            for neighbor in neighbors:
                if self.collision_free(neighbor, point):
                    roadmap.add_edge(
                        point,
                        neighbor,
                        weight=self.two_robots_weight(point, neighbor),
                    )

            if cnt % 100 == 0 and self.verbose:
                print(
                    "connected",
                    cnt,
                    "landmarks to their nearest neighbors",
                    file=self.writer,
                )

    def two_robots_weight(self, p, q):
        """
        Returns weight for edge between p,q which is the sum of distances
        the two robots pass between p,q separately.
        """
        p_list = conversions.Point_d_to_Point_2_list(p)
        q_list = conversions.Point_d_to_Point_2_list(q)
        sum_dist = 0
        for i in range(len(p_list)):
            sum_dist += self.metric.dist(p_list[i], q_list[i]).to_double()
        return sum_dist

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        self.paths_total_length = 0

        # Build collision detection for each robot
        for robot in scene.robots:
            self.collision_detection[robot] = (
                collision_detection.ObjectCollisionDetection(scene.obstacles, robot)
            )

        # Doing `self.num_iterations` separate PRM trials
        roadmaps = []
        self.start = conversions.Point_2_list_to_Point_d(
            [robot.start for robot in scene.robots]
        )
        self.end = conversions.Point_2_list_to_Point_d(
            [robot.end for robot in scene.robots]
        )

        for iter in range(self.num_iterations):
            iter_roadmap = nx.Graph()
            iter_roadmap.add_node(self.start)
            iter_roadmap.add_node(self.end)
            # print("Creating PRM number ", iter, file=self.writer)
            self._create_prm(iter_roadmap)
            roadmaps.append(iter_roadmap)

        # print("", file=self.writer)
        # print("Creating merged roadmap", file=self.writer)

        """
        Extracting best paths from the PRMs
        """
        # The final roadmap built from the points of the best paths
        self.roadmap = nx.Graph()
        paths = []
        for filled_roadmap in roadmaps:
            if nx.algorithms.has_path(filled_roadmap, self.start, self.end):
                tensor_path = nx.algorithms.shortest_path(
                    filled_roadmap, self.start, self.end, weight="weight"
                )
                paths.append(tensor_path)

        # print("Adding points from all the paths", file=self.writer)
        # Adding points from each path to the roadmap
        self.roadmap.add_node(self.start)
        self.roadmap.add_node(self.end)
        for path in paths:
            # first and last points are start and finish
            for point in path[1:-1]:
                self.roadmap.add_node(point)

        # print("Adding possible edges from each path", file=self.writer)
        # Adding possible edges from each path
        for path in paths:
            for i in range(len(path)):
                for j in range(len(path)):
                    if i > j:
                        if self.collision_free(path[i], path[j]):
                            self.roadmap.add_edge(
                                path[j],
                                path[i],
                                weight=self.two_robots_weight(path[j], path[i]),
                            )

        # print("Adding edges from between paths", file=self.writer)
        # Adding edges between paths
        for indx1 in range(len(paths)):
            for indx2 in range(len(paths)):
                if indx1 > indx2:
                    path_1 = paths[indx1]
                    path_2 = paths[indx2]
                    for point_p1 in path_1[1:-1]:
                        for point_p2 in path_2[1:-1]:
                            if self.collision_free(point_p1, point_p2):
                                self.roadmap.add_edge(
                                    point_p2,
                                    point_p1,
                                    weight=self.two_robots_weight(point_p2, point_p1),
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

        # getting total length of paths for the robots
        for i in range(len(tensor_path) - 1):
            self.paths_total_length += self.two_robots_weight(
                tensor_path[i], tensor_path[i + 1]
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

    def get_paths_total_length(self):
        return self.paths_total_length


def experiment_two_robots():
    """
    Used for running the experiments requested in the question.
    """
    import json
    from timeit import default_timer as timer

    from discopygal.solvers import Scene
    from discopygal.solvers.verify_paths import verify_paths

    # from prm import PRM # The original PRM with path distance measurments.

    scene_json = "scenes_2/big_centered_obstacle.json"

    with open(scene_json, "r") as fp:
        scene = Scene.from_dict(json.load(fp))

    NUM_EXPERIMENTS = 10
    NUM_LANDMARKS = 300
    K = 15
    NUM_ITERATIONS = 3

    print(f"{scene_json=} {NUM_EXPERIMENTS=}")

    ########################## Improved PRM ############################

    print(f"Running improved PRM with: {NUM_LANDMARKS=}, {K=}, {NUM_ITERATIONS=}")

    success_counter_improved = 0
    paths_total_length_improved = 0
    start_improved = timer()
    for i in range(NUM_EXPERIMENTS):
        # "Solve" the scene (find paths for the robots)
        solver = PRM_TwoRobots(
            num_landmarks=NUM_LANDMARKS, k=K, num_iterations=NUM_ITERATIONS
        )
        solver.load_scene(scene)
        path_collection = solver.solve()  # Returns a PathCollection object
        paths_total_length_improved += solver.get_paths_total_length()
        # solution = [
        #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
        # ]
        result, reason = verify_paths(scene, path_collection)
        # print(f"Improved Are paths valid: {result}\t{reason}")
        if result:
            success_counter_improved += 1
        # start_gui(scene, solver)
    end_improved = timer()
    average_time_improved = (end_improved - start_improved) / NUM_EXPERIMENTS
    success_rate_improved = (success_counter_improved / NUM_EXPERIMENTS) * 100
    avg_paths_total_length_improved = paths_total_length_improved / NUM_EXPERIMENTS
    print(f"For the improved PRM:")
    print(
        f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time_improved}, success_rate={success_rate_improved}%"
    )
    print(
        f"Average paths length for the two robots = {avg_paths_total_length_improved}"
    )
    print()

    ########################## Basic PRM ############################

    NUM_LANDMARKS_BASIC = NUM_LANDMARKS * NUM_ITERATIONS
    print(f"Running basic PRM with: NUM_LANDMARKS={NUM_LANDMARKS_BASIC}, {K=}")

    success_counter = 0
    paths_total_length = 0
    start = timer()
    for i in range(NUM_EXPERIMENTS):
        # "Solve" the scene (find paths for the robots)
        solver = PRM(num_landmarks=NUM_LANDMARKS_BASIC, k=K)
        solver.load_scene(scene)
        path_collection = solver.solve()  # Returns a PathCollection object
        paths_total_length += solver.get_paths_total_length()
        # solution = [
        #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
        # ]
        result, reason = verify_paths(scene, path_collection)
        # print(f"Improved Are paths valid: {result}\t{reason}")
        if result:
            success_counter += 1
        # start_gui(scene, solver)
    end = timer()
    average_time = (end - start) / NUM_EXPERIMENTS
    success_rate = (success_counter / NUM_EXPERIMENTS) * 100
    avg_paths_total_length = paths_total_length / NUM_EXPERIMENTS
    print(f"For the basic PRM:")
    print(
        f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time}, success_rate={success_rate}%"
    )
    print(f"Average paths length for the two robots = {avg_paths_total_length}")
    print()


# experiment_two_robots()
