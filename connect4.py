import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=Warning)

from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, f1_score

PATH = 'connectfour/'

column_name = []
for i in ['a','b','c','d','e','f','g']:
    for j in range (1,7):
        column_name.append(i+str(j))
        
dic_position = {'x':1,'b':0,'o':-1}
dic_target = {'win':1,'draw':0,'loss':-1}

        
def read_data_and_train(path = PATH):
    df_connect4 = pd.read_csv(path+"connectfour.data", sep = ",", header = None)
    df_connect4.columns = column_name+['y']
    
    for i in column_name:
        df_connect4[i] = df_connect4[i].apply(lambda x : dic_position[x])
    
    df_connect4['y'] = df_connect4['y'].apply(lambda x : dic_target[x])
    
    df_connect4 = df_connect4.sample(frac=1).reset_index(drop=True)
    
    df_connect4_balanced = pd.DataFrame()
    
    df_connect4_1 = df_connect4[df_connect4['y']==1]
    df_connect4_1_ = df_connect4[df_connect4['y']==-1]
    df_connect4_0 = df_connect4[df_connect4['y']==0]
    
    df_connect4_1b = df_connect4_1.sample(len(df_connect4_1_))

    df_connect4_balanced = df_connect4_1b.append(df_connect4_1_).append(df_connect4_0)
    df_connect4_balanced = df_connect4_balanced.sample(frac=1).reset_index(drop=True)
    
    df_connect4_extrapolate = pd.DataFrame()
    
    df_connect4_1 = df_connect4[df_connect4['y']==1]
    df_connect4_1_ = df_connect4[df_connect4['y']==-1]
    df_connect4_0 = df_connect4[df_connect4['y']==0]
    
    df_connect4_1r = df_connect4_1.replace(1,2)
    df_connect4_1r = df_connect4_1r.replace(-1,1)
    df_connect4_1r = df_connect4_1r.replace(2,-1)
    
    df_connect4_1_r = df_connect4_1_.replace(1,2)
    df_connect4_1_r = df_connect4_1_r.replace(-1,1)
    df_connect4_1_r = df_connect4_1_r.replace(2,-1)
    
    df_connect4_0r = df_connect4_0.replace(1,2)
    df_connect4_0r = df_connect4_0r.replace(-1,1)
    df_connect4_0r = df_connect4_0r.replace(2,-1)
    
    df_connect4_extrapolate = pd.concat([df_connect4_1, df_connect4_0, df_connect4_1_,df_connect4_1r, df_connect4_0r, df_connect4_1_r]) 
    df_connect4_extrapolate = df_connect4_extrapolate.drop_duplicates()
    df_connect4_extrapolate = df_connect4_extrapolate.sample(frac=1).reset_index(drop=True)

    
    mlr = MLPRegressor()
    y_pred = cross_val_predict(mlr, df_connect4.drop(['y'], axis = 1), df_connect4[['y']], cv = 5)
    print("r2_score is {}",r2_score(df_connect4['y'], y_pred))
    
    mlr = MLPRegressor()
    y_pred = cross_val_predict(mlr, df_connect4_balanced.drop(['y'], axis = 1), df_connect4_balanced[['y']], cv = 5)
    print("r2_score is {}",r2_score(df_connect4_balanced['y'], y_pred))
    
    mlr = MLPRegressor()
    y_pred = cross_val_predict(mlr, df_connect4_extrapolate.drop(['y'], axis = 1), df_connect4_extrapolate[['y']], cv = 5)
    print("r2_score is {}",r2_score(df_connect4_extrapolate['y'], y_pred))
    
    mlc = MLPClassifier()
    y_pred = cross_val_predict(mlc, df_connect4.drop(['y'], axis = 1), df_connect4[['y']], cv = 5)
    print(confusion_matrix(df_connect4['y'], y_pred, normalize='true'), accuracy_score(df_connect4['y'], y_pred))
    
    mlc = MLPClassifier()
    y_pred = cross_val_predict(mlc, df_connect4_balanced.drop(['y'], axis = 1), df_connect4_balanced[['y']], cv = 5)
    print(confusion_matrix(df_connect4_balanced['y'], y_pred, normalize='true'), accuracy_score(df_connect4_balanced['y'], y_pred))
    
    
    mlc = MLPClassifier((1000, 500, 200, 100, 50, 20), max_iter=5000, early_stopping=True, n_iter_no_change=30)
    mlc.out_activation_ = 'softmax'
    y_pred = cross_val_predict(mlc, df_connect4.drop(['y'], axis = 1), df_connect4[['y']], cv = 5)
    print(confusion_matrix(df_connect4['y'], y_pred, normalize='true'), accuracy_score(df_connect4['y'], y_pred))
    
    mlc = MLPClassifier((1000, 500, 200, 100, 50, 20), max_iter=5000, early_stopping=True, n_iter_no_change=30)
    mlc.out_activation_ = 'softmax'
    y_pred = cross_val_predict(mlc, df_connect4_balanced.drop(['y'], axis = 1), df_connect4_balanced[['y']], cv = 5)
    print(confusion_matrix(df_connect4_balanced['y'], y_pred, normalize='true'), accuracy_score(df_connect4_balanced['y'], y_pred))
    
    
    mlc = MLPClassifier((1000, 500, 200, 100, 50, 20), max_iter=5000, early_stopping=True, n_iter_no_change=30)
    mlc.out_activation_ = 'softmax'
    mlc_train = mlc.fit(df_connect4_balanced.drop(['y'], axis = 1), df_connect4_balanced[['y']])
    
    filename = 'trained_model.pkl'
    pickle.dump(mlc_train, open(filename, 'wb'))
    
    
    
    
def read_trained_model():
    filename = 'trained_model.pkl'
    return pickle.load(open(filename, 'rb'))

def convert_arr_to_model_input(arr):
    dfx = pd.DataFrame(columns = column_name)
    l = []
    for i in range(arr.shape[1]):
        for j in reversed(range(arr.shape[0])):
            l.append(arr[j][i])
    dfx.loc[len(dfx)] = l
    return dfx
    

def print_arr_to_text(arr):
    for i in range(1,8):
        print(str(i),end = " ")
    print("Current Game Positions :")
    for r in arr:
        for c in r:
            d = None
            if c == 0:
                d = '-'
            elif c == 1:
                d = 'x'
            else:
                d = 'o'
            
            print(d,end = " ")
        print()
        
def print_possible_moves(arr):
    l = []
    for i in range(arr.shape[1]):
        if arr[0][i]==0:
            l.append(i+1)
    print("The possible columns we can choose are : ")
    for i in l:
        print(i, end = " ")
    return l

def check_win(arr):
    win = None
    pos = None
    #horizontal check
    for i in range(6):
        for j in range(4):
            if arr[i][j] == arr[i][j+1] == arr[i][j+2] == arr[i][j+3] == 1:
                win = 1
                pos = 'Horizontal Player 1' 
                return win, pos
            if arr[i][j] == arr[i][j+1] == arr[i][j+2] == arr[i][j+3] == -1:
                win = -1
                pos = 'Horizontal Player 2'
                return win, pos
    #vertical check
    for j in range(7):
        for i in range(3):
            if arr[i][j] == arr[i+1][j] == arr[i+2][j] == arr[i+3][j] == 1:
                win = 1
                pos = 'Vertical Player 1'
                return win, pos
            if arr[i][j] == arr[i+1][j] == arr[i+2][j] == arr[i+3][j] == -1:
                win = -1
                pos = 'Vertical Player 2'
                return win, pos
    #diagonal check top left to bottom right
    for row in range(3):
        for col in range(4):
            if arr[row][col] == arr[row + 1][col + 1] == arr[row + 2][col + 2] == arr[row + 3][col + 3] == 1:
                win = 1
                pos = 'Diagonal Player 1'
                return win, pos
            if arr[row][col] == arr[row + 1][col + 1] == arr[row + 2][col + 2] == arr[row + 3][col + 3] == -1:
                win = -1
                pos = 'Diagonal Player 2'
                return win, pos
    
    #diagonal check bottom left to top right
    for row in range(5, 2, -1):
        for col in range(3):
            if arr[row][col] == arr[row - 1][col + 1] == arr[row - 2][col + 2] == arr[row - 3][col + 3] == 1:
                win = 1
                pos = 'Diagonal Player 1'
                return win, pos
            if arr[row][col] == arr[row - 1][col + 1] == arr[row - 2][col + 2] == arr[row - 3][col + 3] == -1:
                win = -1
                pos = 'Diagonal Player 2'
                return win, pos
                
    #check tie
    t = 1
    for col in range(7):
        t = t*arr[0][col]
    if t!=0:
        win = 0
        pos = 'Tie Player 1 and Player 2'
                
    return win, pos
    

def print_state_after_move(arr, move, player):
    flag = 0
    while flag==0:
        for row in reversed(range(arr.shape[0])):
            if arr[row][move-1]==0:
                arr[row][move-1] = player
                flag = 1
                break
    
    print_arr_to_text(arr)
    win, pos = check_win(arr)
    return arr, win, pos

def get_model_inputs(arr, l):
    dff = pd.DataFrame()
    for i in l:
        arr1 = np.array(arr)
        flag = 0
        while flag==0:
            for row in reversed(range(arr1.shape[0])):
                if arr1[row][i-1]==0:
                    arr1[row][i-1] = player
                    flag = 1
                    break
        dfx = convert_arr_to_model_input(arr1)
        dff = dff.append(dfx)
    dff = dff.reset_index(drop = True)
    return dff
        

def algo_move(arr, player_turn, l, model):
    dff = get_model_inputs(arr, l)
    dff_pred = pd.DataFrame(model.predict_proba(dff), columns = model.classes_)
    
    if player_turn == 1:
        dff_pred = dff_pred.sort_values(by = [-1], ascending = False)
    else:
        dff_pred = dff_pred.sort_values(by = [1], ascending = False)
    return l[dff_pred.index[0]]


#read_data_and_train()
model = read_trained_model()
arr = np.random.randint(low = 0, high = 1, size = (6,7))
win = None
pos = None
player = 1
move_num = 0

print(" Hi!! My name is Anticipator. Lets see if I can live upto my name. I'm extremely courteous.")
print("Select if you want to go first or second. Please enter 1 or 2 accordingly.")

player_turn = input()

while type(player_turn)!=int:
    try:
        player_turn = int(player_turn)
        if player_turn not in [1,2]:
            print("Please enter either 1 or 2 and press ENTER/RETURN")
            player_turn = input()
    except:
        print("Enter an integer than is 1 or 2 and press ENTER/RETURN")
        player_turn = input()
        
if player_turn ==1:
    print("Your moves will be represented as x , mine will be represented as o and an empty slot as b")
else:
    print("Your moves will be represented as o , mine will be represented as x and an empty slot as b")
        

while win==None:
    
    l = print_possible_moves(arr)
    
    if move_num%2 == (player_turn-1)%2:
        print("Enter the column you wish to choose and press ENTER/RETURN")
        move = input()
        while type(move)!=int:
            try:
                move = int(move)
                if move not in l:
                    if move in range (1,8):
                        print("This column is full. Enter an acceptable column and press ENTER/RETURN")
                    else:
                        print("This column is not available. Enter an acceptable column and press ENTER/RETURN")
                    move = input()
            except:
                print("Enter an integer and press ENTER/RETURN")
                move = input()
    else:
        move = algo_move(arr, player_turn, l, model)
        print("The column I pick is ", move)
        
    arr, win, pos = print_state_after_move(arr, move, player)
    player = player*(-1)
    
    move_num = move_num+1
    
if win == 1:
    if player_turn ==1:
        print(pos+' Wins')
        print("you WIN!!! Looks like I'm not the anticipator afterall")
    else:
        print(pos+' Wins')
        print("I WINN!! Better Luck next time.")
elif win == -1:
    if player_turn ==2:
        print(pos+' Wins')
        print("you WIN!!! Looks like I'm not the anticipator afterall")
    else:
        print(pos+' Wins')
        print("I WINN!! Better Luck next time.")
else:
    print("Its a TIE!!!")
    
