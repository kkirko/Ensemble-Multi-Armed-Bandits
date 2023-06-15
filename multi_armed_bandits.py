import numpy as np
from sklearn.metrics import f1_score

class MultiArmedBandits:
    # Initialize the total rewards, counts and average rewards for all models
    def __init__(self, models):
        self.models = models
        self.total_rewards = np.zeros(len(models))
        self.counts = np.zeros(len(models))
        self.average_rewards = np.zeros(len(models))

    # Select a model for the next step
    # With probability `epsilon`, select a model randomly (exploration)
    # With probability `1 - epsilon`, select the model that has the highest average reward so far (exploitation)    
    def select_model(self, epsilon):
        rand_num = np.random.random()
        if rand_num < epsilon:
            return np.random.randint(len(self.models))
        else:
            return np.argmax(self.average_rewards)

    # Update the total reward, count and average reward for the selected model
    def update_reward(self, model_index, reward):
        self.counts[model_index] += 1
        self.total_rewards[model_index] += reward
        self.average_rewards[model_index] = self.total_rewards[model_index] / self.counts[model_index]

# Define the function for model selection using multi-armed bandits
def bandit_model_selection(models, X_test, y_test, epsilon=0.2, batch_size=150):
    bandits = MultiArmedBandits(models)
    best_model_rewards = []
    model_choice_at_steps = []

    # Process the test data in batches
    for i in range(0, len(X_test), batch_size):
        # Get the current batch of data
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]

        # Select a model
        model_index = bandits.select_model(epsilon)
        # Make a prediction and get the reward
        y_pred = models[model_index].predict(X_batch)
        reward = f1_score(y_batch, y_pred)

        # Update the rewards for the selected model
        bandits.update_reward(model_index, reward)

        # Get the best model so far and its reward
        best_model_index = np.argmax(bandits.average_rewards)
        best_model_rewards.append(bandits.average_rewards[best_model_index])
        model_choice_at_steps.append(model_index)

    return np.array(best_model_rewards), np.array(model_choice_at_steps)
