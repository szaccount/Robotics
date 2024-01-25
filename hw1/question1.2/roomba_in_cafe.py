import json

from discopygal.solvers import Scene
from discopygal.solvers.verify_paths import verify_paths
from discopygal_tools.solver_viewer import start_gui

from RoombaSolver import RoombaSolver

with open("coffee_shop.json", "r") as fp:
    scene = Scene.from_dict(json.load(fp))

# "Solve" the scene (find paths for the robots)
solver = RoombaSolver(sample_points=2000, num_neighbors=15)
solver.load_scene(scene)
path_collection = solver.solve()  # Returns a PathCollection object

# Print the points of the paths
for i, (robot, path) in enumerate(path_collection.paths.items()):
    print("Path for robot {}:".format(i))
    for j, point in enumerate(path.points):
        print(
            f"\t Point {j:2}:  ", point.location
        )  # point is of type PathPoint, point.location is CGALPY.Ker.Point_2
    print()

result, reason = verify_paths(scene, path_collection)
print(f"Are paths valid: {result}\t{reason}")

start_gui(scene, solver)
