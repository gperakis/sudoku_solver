import argparse
from sudoku_solver.solve import SudokuSolver

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each "
                     "step of the pipeline")
args = vars(ap.parse_args())

if __name__ == "__main__":
    # USAGE
    # python -m cli --image sudoku_puzzle.jpg --debug 1
    solver = SudokuSolver()
    solver.solve(image_path=args['image'],
                 debug=args['debug'])
