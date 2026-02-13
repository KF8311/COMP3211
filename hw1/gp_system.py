import numpy as np

epoches = 100  # 3.6 hardcoded number of iterations
SIZE = 1000  # number of programs

def predict(programs, X):
    scores = X @ programs.T  # predict the result with np operations
    return (scores > 0).astype(int)  # score more than 0 than predicted 1, 0 otherwise

def fitness(predictions, label):
    # 3.1 fitness function
    result = (
        predictions == label[:, None]
    )  # when equal, prediction is correct, award one score
    return result.mean(
        axis=0
    )  # calculate the accuracy with correct_sample/total_sample

def mutate(x1):
    # 3.4 mutation operator
    num = np.random.randint(0, len(x1))  # select a random index
    x1[num] = 2 * np.random.rand() - 1  # new random float in (-1,1) return x1
    return x1

# convert the csv to a numpy array
data = np.loadtxt("gp-training-set.csv", delimiter=",")
X = data[:, :-1]  # features value of all parameters from x1 to xn
y = data[:, -1]  # binary label of test data
x_length = len(X[0])  # total number of the parameters

# 3.5 initial generation
programs = (
    2 * np.random.rand(SIZE, x_length) - 1
)  # generate SIZE random programs for generation 0

for k in range(epoches):
    # select half of the programs and split into tournament groups
    selected_indices = np.random.choice(SIZE, size=SIZE // 2, replace=False)
    selected = programs[selected_indices]
    np.random.shuffle(selected)  # shuffle them into different groups
    pred = predict(selected, X)  # evaluate those programs
    scores = fitness(pred, y)  # measure their accuracy with pred

    # adaptively split selected into up to 10 groups
    num_selected = selected.shape[0]
    num_groups = min(10, num_selected)
    selected_groups = np.array_split(selected, num_groups)
    score_groups = np.array_split(scores, num_groups)
    winners_list = []  # 3.3 copy operator, we copy the winners
    for g, sg in zip(selected_groups, score_groups):
        # pick the best program in each group
        winner_idx = np.argmax(sg)
        winners_list.append(g[winner_idx])
    winners = np.stack(winners_list)  # put the winner

    # select remaining programs for crossover to fill the population
    num_children = max(
        0, SIZE - len(winners)
    )  # generate the remaining programs with crossover
    crossover_indices = np.random.choice(SIZE, size=(num_children, 2), replace=True)
    parents1 = programs[crossover_indices[:, 0]]  # parent 1 for crossover
    parents2 = programs[crossover_indices[:, 1]]  # parent 2 for crossover
    # 3.2 crossover operator
    mask = (
        np.random.rand(*parents1.shape) > 0.5
    )  # random generate 1s and 0s to select crossover parameters from x1 or x2
    children = np.where(
        mask, parents1, parents2
    )  # mask the binary to the arrays and combine them
    programs = np.vstack([winners, children])  # Combine winners and children
    programs = programs[:SIZE]  # restrict the program size
    mutate_index = np.random.randint(0, SIZE)  # randomly pick 1 program to mutate
    programs[mutate_index] = mutate(programs[mutate_index])

# find the best program after finish the training
final_pred = predict(programs, X)  # evaluate those final programs
final_scores = fitness(final_pred, y)  # measure their accuracy with pred
best_index = np.argmax(final_scores)  # select the highest accuracy program
best_program = programs[best_index]  # best program
best_score = final_scores[best_index]  # accuracy of best program
print("Prediction of parameters: ", best_program)  # Show the result
print("Accuracy: ", best_score)
