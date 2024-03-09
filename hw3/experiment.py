import json
from timeit import default_timer as timer

from discopygal.solvers import Scene
from discopygal.solvers.verify_paths import verify_paths

# from discopygal_tools.solver_viewer import start_gui

from prm_rod import BasicRodPRM
from prm_rod_improved import ImprovedRodPRM


scene_json = "rod_scenes/1_rod_example.json"

with open(scene_json, "r") as fp:
    scene = Scene.from_dict(json.load(fp))

NUM_EXPERIMENTS = 2
NUM_LANDMARKS = 400
K = 15

success_counter = 0
start = timer()
for i in range(NUM_EXPERIMENTS):
    # "Solve" the scene (find paths for the robots)
    solver = BasicRodPRM(num_landmarks=NUM_LANDMARKS, k=K)
    solver.load_scene(scene)
    path_collection = solver.solve()  # Returns a PathCollection object

    # solution = [
    #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
    # ]

    result, reason = verify_paths(scene, path_collection)
    print(f"Are paths valid: {result}\t{reason}")
    if result:
        success_counter += 1

    # start_gui(scene, solver)
end = timer()

average_time = (end - start) / NUM_EXPERIMENTS
success_rate = (success_counter / NUM_EXPERIMENTS) * 100
print("For the basic PRM:")
print(
    f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time}, success_rate={success_rate}%"
)

success_counter_improved = 0
start_improved = timer()
for i in range(NUM_EXPERIMENTS):
    # "Solve" the scene (find paths for the robots)
    solver = ImprovedRodPRM(num_landmarks=NUM_LANDMARKS, k=K, gaussian_ratio=0.9)
    solver.load_scene(scene)
    path_collection = solver.solve()  # Returns a PathCollection object

    # solution = [
    #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
    # ]

    result, reason = verify_paths(scene, path_collection)
    print(f"Improved Are paths valid: {result}\t{reason}")
    if result:
        success_counter_improved += 1

    # start_gui(scene, solver)
end_improved = timer()

average_time_improved = (end_improved - start_improved) / NUM_EXPERIMENTS
success_rate_improved = (success_counter_improved / NUM_EXPERIMENTS) * 100
print("For the improved PRM:")
print(
    f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time_improved}, success_rate={success_rate_improved}%"
)
