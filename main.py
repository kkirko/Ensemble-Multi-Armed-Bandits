from generate_data import generate_data
from train_models import train_models
from multi_armed_bandits import bandit_model_selection

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Generate data
X_train, X_test, y_train, y_test = generate_data()

# Classifiers training
catboost_clf, lightgbm_clf, random_forest_clf = train_models(X_train, y_train)

# Average F1 Score
avg_pred = (catboost_clf.predict_proba(X_test) * 1/3) + (lightgbm_clf.predict_proba(X_test) * 1/3) + (random_forest_clf.predict_proba(X_test) * 1/3)
avg_pred = (avg_pred > 0.5).astype(int)
avg_f1_score = f1_score(y_test, avg_pred[:, 1])

# Bandit model selection
models = [catboost_clf, lightgbm_clf, random_forest_clf]
best_model_rewards, model_choices = bandit_model_selection(models, X_test, y_test)

# Plot
plt.figure(figsize=(20, 8))
for i in range(len(best_model_rewards) - 1):
    color = 'b' if model_choices[i] == 0 else 'r' if model_choices[i] == 1 else 'g'
    plt.plot(range(i, i+2), best_model_rewards[i:i+2], color)

plt.axhline(y=avg_f1_score, color='c', linestyle='--')
plt.xlabel("Batch")
plt.ylabel("Best Model Average Reward")

blue_patch = mpatches.Patch(color='blue', label='catboost')
red_patch = mpatches.Patch(color='red', label='lightgbm')
green_patch = mpatches.Patch(color='green', label='random_forest')
cyan_patch = mpatches.Patch(color='cyan', label='average f1_score')
plt.legend(handles=[blue_patch, red_patch, green_patch, cyan_patch])
plt.show()