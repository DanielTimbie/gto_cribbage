import numpy as np


action_to_take = np.zeros(4)
action_to_take[2] = 1
print(action_to_take)

print(np.argmax(action_to_take))