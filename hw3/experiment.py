import json

from discopygal.solvers import Scene
from discopygal.solvers.verify_paths import verify_paths

# from discopygal_tools.solver_viewer import start_gui

from prm_rod_improved import ImprovedRodPRM

scene_json = "rod_scenes/1_rod_example.json"

with open(scene_json, "r") as fp:
    scene = Scene.from_dict(json.load(fp))

for i in range(10):
    # "Solve" the scene (find paths for the robots)
    solver = ImprovedRodPRM(num_landmarks=500, k=15, gaussian_ratio=0.9)
    solver.load_scene(scene)
    path_collection = solver.solve()  # Returns a PathCollection object

    # solution = [
    #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
    # ]

    result, reason = verify_paths(scene, path_collection)
    print(f"Are paths valid: {result}\t{reason}")

    # start_gui(scene, solver)
