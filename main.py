from dnc import SAM
from torch.optim import Adam
from torch.nn import cross_entropy
from torch import save, load

#network parameters
network = SAM(
	input_size=64,
	hidden_size=128,
	rnn_type="lstm",
	num_layers=4,
	nr_cells=5000,
	cell_size=32,
	read_heads=4,
	sparse_reads=4,
	batch_first=True,
	gpu_id=0
)

#configuration variables
num_epochs = 100000
learning_rate = 1e-4
SAVE_PATH = "/saves/model"

def train():
	#inital values for SAM's hidden layer
	hidden = (None, None, None)

	#initialize the optimizer
	optimizer = Adam(network.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		### FORWARD PASS ###
		output, hidden, debug = network(inputs, hidden, reset_experience=True)

		### BACKWARD PASS ###
		#zero gradient before backwards pass
		optimizer.zero_grad()

		#compute loss and then compute gradient
		loss = cross_entropy(output, target)
		loss.backwards()

		#update weights using optimizer and loss
		optimizer.step()

		#print training status
		print("Epoch {} Loss: {}".format(epoch, loss.item()))

	#save the model
	torch.save(network.state_dict(), SAVE_PATH)

def test():
	try:
		network.load_state_dict(torch.load(SAVE_PATH))
	except:
		print("ERROR! You most likely tried to test the network without loading a model. Check that there is a saved model in {}".format(SAVE_PATH))