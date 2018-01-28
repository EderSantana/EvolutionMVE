import json
import numpy as np
from tensorflow import keras
Sequential =  keras.models.Sequential
Dense = keras.layers.Dense
sgd = keras.optimizers.SGD


class Catch(object):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return 0 
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]

 
def play_round(model, grid_size):
    # Define environment/game
    env = Catch(grid_size)
    input_t = env.observe()
    game_over = False

    while not game_over:
        input_tm1 = input_t
        # get next action
        q = model.predict(input_tm1)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        input_t, reward, game_over = env.act(action)

    return reward


if __name__ == "__main__":
    # parameters
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 5000  # how many runs 
    hidden_size = 100
    population_size = 100  # how many different random agents per run
    learning_rate = .1
    sigma = .5  # std of the noise that defines weight perturbation (with large sigma you explore further, but may never really get anywhere)
    grid_size = 10
    games_per_agent = 5

    # this model is smaller than the Q-learning model (see the comment oriented programming, COP, down there?)
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    #model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow (I love COP)
    # model.load_weights("model.h5")

    # Train
    for e in range(epoch):
        weights = model.get_weights()
        all_eps = []
        rewards = np.zeros(population_size)
        for i, p in enumerate(range(population_size)):
            new_weights = []
            eps = []
            for w in weights:
                eps.append(np.random.randn(*w.shape))
                new_weights.append(w + sigma * eps[-1])
            all_eps.append(eps)
            model.set_weights(new_weights) 
            for j in range(games_per_agent):
                rewards[i] += play_round(model, grid_size)

        # total reward of everything explored
        win_cnt = rewards.sum()
        reward_norm = (rewards - rewards.mean()) / rewards.std()

        # correlate reward with noise and update weights
        updated_weights = []
        for index, w in enumerate(weights):
            X = np.array([eps[index] for eps in all_eps])  # all noise added to weight[index]
            w = w + learning_rate / (population_size * sigma) * X.T.dot(reward_norm).T
            updated_weights.append(w)

        # your new model is good to go
        model.set_weights(updated_weights)

        # decay learning rate
        # learning_rate *= .993

        print("Epoch {:03d}/{:04d} | Win count {}".format(e, epoch-1, win_cnt))
             
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
