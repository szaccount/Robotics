import json
from timeit import default_timer as timer

from discopygal.solvers import Scene
from discopygal.solvers.verify_paths import verify_paths

# from prm_rod import BasicRodPRM
from prm_rod_improved import ImprovedRodPRM

from rrt_rod import RodRRT


scene_json = "rod_scenes/custom_4_more_densely_populated.json"

with open(scene_json, "r") as fp:
    scene = Scene.from_dict(json.load(fp))

NUM_EXPERIMENTS = 10
NUM_LANDMARKS = 100
K = 15
ETA = 1

print(f"{scene_json=}, {NUM_EXPERIMENTS=}, {NUM_LANDMARKS=}, {K=}, {ETA=}")

########################## Basic PRM ############################

# success_counter = 0
# start = timer()
# for i in range(NUM_EXPERIMENTS):
#     # "Solve" the scene (find paths for the robots)
#     solver = BasicRodPRM(num_landmarks=NUM_LANDMARKS, k=K)
#     solver.load_scene(scene)
#     path_collection = solver.solve()  # Returns a PathCollection object

#     # solution = [
#     #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
#     # ]

#     result, reason = verify_paths(scene, path_collection)
#     # print(f"Are paths valid: {result}\t{reason}")
#     if result:
#         success_counter += 1

#     # start_gui(scene, solver)
# end = timer()

# average_time = (end - start) / NUM_EXPERIMENTS
# success_rate = (success_counter / NUM_EXPERIMENTS) * 100
# print("For the basic PRM:")
# print(
#     f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time}, success_rate={success_rate}%"
# )
# print()

########################## RRT ############################

success_counter_rrt = 0
start_rrt = timer()
for i in range(NUM_EXPERIMENTS):
    # "Solve" the scene (find paths for the robots)
    solver = RodRRT(num_landmarks=NUM_LANDMARKS, eta=ETA)
    solver.load_scene(scene)
    path_collection = solver.solve()  # Returns a PathCollection object

    # solution = [
    #     (robot, path) for i, (robot, path) in enumerate(path_collection.paths.items())
    # ]

    result, reason = verify_paths(scene, path_collection)
    # print(f"Are paths valid: {result}\t{reason}")
    if result:
        success_counter_rrt += 1

    # start_gui(scene, solver)
end_rrt = timer()

average_time_rrt = (end_rrt - start_rrt) / NUM_EXPERIMENTS
success_rate_rrt = (success_counter_rrt / NUM_EXPERIMENTS) * 100
print("For RRT:")
print(
    f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time_rrt}, success_rate={success_rate_rrt}%"
)
print()

########################## Improved PRM ############################

for gaus_ratio in [0.2, 0.8, 1]:
    success_counter_improved = 0
    start_improved = timer()
    for i in range(NUM_EXPERIMENTS):
        # "Solve" the scene (find paths for the robots)
        solver = ImprovedRodPRM(
            num_landmarks=NUM_LANDMARKS, k=K, gaussian_ratio=gaus_ratio
        )
        solver.load_scene(scene)
        path_collection = solver.solve()  # Returns a PathCollection object

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
    print(f"For the improved PRM with {gaus_ratio=}:")
    print(
        f"Did {NUM_EXPERIMENTS} experiments, average_time={average_time_improved}, success_rate={success_rate_improved}%"
    )
    print()
