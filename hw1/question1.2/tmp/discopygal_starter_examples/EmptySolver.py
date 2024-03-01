from discopygal.solvers.Solver import Solver


class EmptySolver(Solver):
    def solve(self):
        self.log("Solving...")
        return None

    def load_scene(self, scene):
        pass

    @staticmethod
    def get_arguments():
        return {}
