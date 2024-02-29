import json

from discopygal.bindings import *
from discopygal.solvers import PathPoint, Scene, RobotDisc, ObstaclePolygon
from discopygal.solvers.rrt import dRRT
from discopygal.solvers.verify_paths import verify_paths
from discopygal_tools.solver_viewer import start_gui

if __name__ == "__main__":
    scene = Scene()

    # Illustration of the scene:
    # Two disc robots in a room, with an obstacle separating them
    # The goal of the robots is to switch positions
    ###############################
    #                             #
    #                             #
    #                             #
    #      0      ###      0      #
    #     000     ###     000     #
    #      0      ###      0      #
    #                             #
    #                             #
    #                             #
    ###############################

    # Add disc robot on the left that wants to go to the right
    left_robot = RobotDisc(
        radius=FT(1.0), 
        start=Point_2(FT(-3.0), FT(0.0)),
        end=Point_2(FT(3.0), FT(0.0)))
    right_robot = RobotDisc(
        radius=FT(1.0), 
        start=Point_2(FT(3.0), FT(0.0)),
        end=Point_2(FT(-3.0), FT(0.0)))
    scene.add_robot(left_robot)
    scene.add_robot(right_robot)

    # Add obstacle in the middle
    middle_obstacle = ObstaclePolygon(
        poly=Polygon_2([
            Point_2(FT(-1), FT(-1)),
            Point_2(FT(-1), FT(1)),
            Point_2(FT(1), FT(1)),
            Point_2(FT(1), FT(-1))
        ])
    )
    scene.add_obstacle(middle_obstacle)

    # Also add bounding walls
    left_wall = ObstaclePolygon(
        poly=Polygon_2([
            Point_2(FT(-7), FT(-4)),
            Point_2(FT(-7), FT(4)),
            Point_2(FT(-6), FT(4)),
            Point_2(FT(-6), FT(-4))
        ])
    )
    right_wall = ObstaclePolygon(
        poly=Polygon_2([
            Point_2(FT(6), FT(-4)),
            Point_2(FT(6), FT(4)),
            Point_2(FT(7), FT(4)),
            Point_2(FT(7), FT(-4))
        ])
    )
    top_wall = ObstaclePolygon(
        poly=Polygon_2([
            Point_2(FT(-6), FT(3.5)),
            Point_2(FT(-6), FT(4)),
            Point_2(FT(6), FT(4)),
            Point_2(FT(6), FT(3.5))
        ])
    )
    bottom_wall = ObstaclePolygon(
        poly=Polygon_2([
            Point_2(FT(-6), FT(-4)),
            Point_2(FT(-6), FT(-3.5)),
            Point_2(FT(6), FT(-3.5)),
            Point_2(FT(6), FT(-4))
        ])
    )
    scene.add_obstacle(left_wall)
    scene.add_obstacle(right_wall)
    scene.add_obstacle(top_wall)
    scene.add_obstacle(bottom_wall)

    # Export the scene to file
    with open('simple_motion_planning_scene.json', 'w') as fp:
        json.dump(scene.to_dict(), fp)

    # "Solve" the scene (find paths for the robots)
    solver = dRRT(num_landmarks=100, prm_num_landmarks=200, prm_k=15, bounding_margin_width_factor=0)
    solver.load_scene(scene)
    path_collection = solver.solve() # Returns a PathCollection object

    # Print the points of the paths
    for i, (robot, path) in enumerate(path_collection.paths.items()):
        print("Path for robot {}:".format(i))
        for point in path.points:
            print('\t', point.location) # point is of type PathPoint, point.location is CGALPY.Ker.Point_2

    result, reason = verify_paths(scene, path_collection)
    assert result  # Validate path if is valid
    print(f"Are paths valid: {result}")

    # Start solver_viewer with the scene and solver objects (the scene isn't solved yet)
    start_gui(scene, solver)
    print("Ended gui")