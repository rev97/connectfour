import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

from utils import get_data, check_game_over, encode_board
# Define command-line arguments
parser = argparse.ArgumentParser(description='Connect Four gameplay')
parser.add_argument('--data-dir', type=str, default=r'/Users/revanthgottuparthy/Desktop/my projects/connectfour/connectfour/connectfour.txt', 
                    help='Root directory containing class folders')
parser.add_argument('--num-epochs', type=int, default=10, help='model used for bot')
args = parser.parse_args()


connect_four_data = get_data(args.data_dir)
# Preprocess the data
X = [row.split(',')[:-1] for row in connect_four_data]
y = [row.split(',')[-1] for row in connect_four_data]



# Convert board positions to numerical features using LabelEncoder
encoder = LabelEncoder()
X_encoded = [encoder.fit_transform(row) for row in X]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
'''
# Train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
'''
# Train a Neural Network model (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=args.num_epochs, random_state=42)
model.fit(X_train, y_train)
# Function to display the game board
def display_board(board):
    for row in reversed(range(6)):
        print(" ".join(board[col][row] for col in range(7)))
    print("a b c d e f g")

# Initialize an empty game board
board = [["."] * 6 for _ in range(7)]

# Mapping column labels to indices
column_labels = "abcdefg"
col_indices = {label: index for index, label in enumerate(column_labels)}

# Game loop
player_turn = True  # True for human, False for computer
while True:
    display_board(board)
    
    if player_turn:
        valid_move = False
        while not valid_move:
            player_col = input("Your move - Column (a-g): ")
            if player_col in col_indices:
                col_index = col_indices[player_col]
                if "." in board[col_index]:
                    player_row = int(input("Your move - Row (1-6): ")) - 1
                    if 0 <= player_row < 6 and board[col_index][player_row] == ".":
                        valid_move = True
                        board[col_index][player_row] = "x"
                    else:
                        print("Invalid row. Choose an empty row (1-6).")
                else:
                    print("Column full. Choose another column.")
            else:
                print("Invalid column. Choose a valid column (a-g).")
    else:
        # Computer's move using the trained model
        available_moves = [(col_index, row_index) for col_index, col in enumerate(board) for row_index, cell in enumerate(col) if cell == "."]
        
        # Initialize available_moves_encoded with flattened board for each possible move
        available_moves_encoded = []
        for col_index, row_index in available_moves:
            temp_board = [row[:] for row in board]  # Create a copy of the board
            temp_board[col_index][row_index] = "o"
            available_moves_encoded.append(encode_board(temp_board).flatten())
        
        # Predict probabilities for all possible moves
        predicted_probabilities = model.predict_proba(available_moves_encoded)
        
        # Find the best move based on highest predicted probability of "o" move
        best_move_index = np.argmax(predicted_probabilities[:, 2])  # Choose the index with highest "o" probability
        print(best_move_index)
        best_col_index, best_row_index = available_moves[best_move_index]
        board[best_col_index][best_row_index] = "o"
    
    if all("." not in col for col in board) or check_game_over(board, "x") or check_game_over(board, "o"):
        display_board(board)
        if check_game_over(board, "o"):
            print("Computer wins!")
        elif check_game_over(board, "x"):
            print("Player wins!")
        else:
            print("It's a draw!")
        break
    
    player_turn = not player_turn
