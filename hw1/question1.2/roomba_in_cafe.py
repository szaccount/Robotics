import json

from discopygal.solvers import Scene
from discopygal.solvers.verify_paths import verify_paths
from discopygal_tools.solver_viewer import start_gui

from RoombaSolver import RoombaSolver

with open("coffee_shop.json", "r") as fp:
    scene = Scene.from_dict(json.load(fp))

# "Solve" the scene (find paths for the robots)
solver = RoombaSolver(sample_points=5000, samples_for_edge=10, num_neighbors=15)
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

# Optional - Open solver_viewer with our solver and scene

# Option 1 - Use solve object we made:
print("First gui")
start_gui(scene, solver)

# # Option 2 - Use solver's class type to create a new one:
# print("Second gui")
# start_gui(scene, RoombaSolver)

# # Option 3 - Use solver's class name to create a new one (must pass solver's module):
# print("Third gui")
# start_gui(scene, "RoombaSolver", "RoombaSolver.py")

# # Option 4 - Passing the path to the scene json file
# print("Fourth gui")
# start_gui("coffee_shop.json", solver)
