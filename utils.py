import numpy as np
def get_data(file_path):
    # Read the lines from the text file
    with open(file_path, "r") as file:
        connect_four_data = file.read().strip().split('\n')
    return connect_four_data

def check_game_over(board, symbol):
    # Check rows
    for row in range(6):
        for col in range(4):
            if all(board[col+i][row] == symbol for i in range(4)):
                return True
    
    # Check columns
    for col in range(7):
        for row in range(3):
            if all(board[col][row+i] == symbol for i in range(4)):
                return True
    
    # Check diagonals
    for col in range(4):
        for row in range(3):
            if all(board[col+i][row+i] == symbol for i in range(4)):
                return True
            if all(board[col+i][row+3-i] == symbol for i in range(4)):
                return True
    
    return False

def encode_board(board):
    mapping = {'.': 0, 'x': 1, 'o': 2}
    return np.array([[mapping[cell] for cell in col] for col in board])