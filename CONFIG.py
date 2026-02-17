import numpy as np

possible_actions = {
	"train": "train",
	"validate": "validate",
}
possible_actions_list = list(possible_actions.keys())

############################
#	NEURAL PARAMETERS
############################

img_name = "me.jpg"
model_name = img_name + ".pt"

learning_rate = 1e-2
epochs = 2000
nodes = 32
depth = 16

sample_decay = 10

batch_size = 4096 * 2 * 2 * 2 * 2

print_every_epoch = 10