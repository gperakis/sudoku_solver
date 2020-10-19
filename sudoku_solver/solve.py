import cv2
import imutils
import numpy as np
from sudoku import Sudoku
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Optional
from sudoku_solver import extract_digit, find_puzzle


class SudokuSolver:
    def __init__(self):
        self._model: Optional[Sequential] = None

    @property
    def model(self) -> Sequential:
        if self._model is None:
            # load the digit classifier from disk
            print("[INFO] loading digit classifier...")
            self._model = load_model('output/digit_classifier.h5')

        return self._model

    @staticmethod
    def load_image(image):
        # load the input image from disk and resize it
        print("[INFO] processing image...")
        image = cv2.imread(image)
        image = imutils.resize(image, width=600)
        return image

    def solve(self, image_path, debug):
        image = self.load_image(image_path)

        # find the puzzle in the image and then
        (puzzle_image, warped) = find_puzzle(image, debug=debug > 0)

        # initialize our 9x9 sudoku board
        board = np.zeros((9, 9), dtype="int")

        # a sudoku puzzle is a 9x9 grid (81 individual cells), so we can infer
        # the location of each cell by dividing the warped image into a
        # 9x9 grid
        step_x = warped.shape[1] // 9
        step_y = warped.shape[0] // 9

        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cell_locations = []

        # loop over the grid locations
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []

            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                start_x = x * step_x
                start_y = y * step_y
                end_x = (x + 1) * step_x
                end_y = (y + 1) * step_y

                # add the (x, y)-coordinates to our cell locations list
                row.append((start_x, start_y, end_x, end_y))

                # crop the cell from the warped transform image and then
                # extract the digit from the cell
                cell = warped[start_y:end_y, start_x:end_x]
                digit = extract_digit(cell, debug=debug > 0)

                # verify that the digit is not empty
                if digit is not None:
                    foo = np.hstack([cell, digit])
                    cv2.imshow("Cell/Digit", foo)

                    # resize the cell to 28x28 pixels and then prepare the
                    # cell for classification
                    roi = cv2.resize(digit, (28, 28))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the digit and update the sudoku board with the
                    # prediction
                    pred = self.model.predict(roi).argmax(axis=1)[0]
                    board[y, x] = pred

            # add the row to our cell locations
            cell_locations.append(row)

        # construct a sudoku puzzle from the board
        print("[INFO] OCR'd sudoku board:")
        puzzle = Sudoku(3, 3, board=board.tolist())
        puzzle.show()

        # solve the sudoku puzzle
        print("[INFO] solving sudoku puzzle...")
        solution = puzzle.solve()
        solution.show_full()

        # loop over the cell locations and board
        for (cellRow, boardRow) in zip(cell_locations, solution.board):
            # loop over individual cell in the row
            for (box, digit) in zip(cellRow, boardRow):
                # unpack the cell coordinates
                start_x, start_y, end_x, end_y = box

                # compute the coordinates of where the digit will be drawn
                # on the output puzzle image
                textX = int((end_x - start_x) * 0.33)
                textY = int((end_y - start_y) * -0.2)
                textX += start_x
                textY += end_y

                # draw the result digit on the sudoku puzzle image
                cv2.putText(puzzle_image, str(digit), (textX, textY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # show the output image
        cv2.imshow("Sudoku Result", puzzle_image)
        cv2.waitKey(0)
