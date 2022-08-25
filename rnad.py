from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

size = 3
#global hyper parameter


################


class Net (torch.nn.Module):

    def __init__ (self):
        super().__init__()

        self.fc0 = nn.Linear(size**2, size**2)
        self.fc1 = nn.Linear(size**2, size)
        self.fc2 = nn.Linear(size**2, 1)

    def forward (self, input):
        tower = torch.relu(self.fc0(input.view(-1, size**2)))
        logits = self.fc1(tower)
        policy = F.softmax(logits, dim=1)
        value = torch.tanh(self.fc2(tower))
        return logits, policy, value


################

class RNaD:
# container class for RNaD on the one-shot matrix game example

    def __init__(self):

        self.illustrative_example = None
        self.batch_size = 2**10
        self.validation_batch = torch.normal(0, 1, size=(self.batch_size, size, size))

        self.net = Net()
        self.net_checkpoints = {}

        self.gamma = .1
        self.total_NeuRD_steps = 2**10
        self.total_RNaD_steps = 2**10

    def generate_input_batch (self, batch_size : int, matrix_entry_distribution=None) -> torch.tensor :

        if matrix_entry_distribution is None:
            matrix_entry_distribution = torch.Generator() #fix so zero mean

        #player_one_batch = torch.rand(size=(batch_size, size, size), generator=matrix_entry_distribution)
        player_one_batch = torch.normal(0, 1, size=(self.batch_size, size, size))
        player_two_batch = player_one_batch.clone().detach()
        player_two_batch = -player_two_batch.swapaxes(1, 2) #zero sum
        input_batch = torch.cat((player_one_batch, player_two_batch), dim=0)
        return input_batch

    def exploitability (self, input, player_one_strategies, players_two_strategies):

        pass

    def compute_loss_from_PG_step (self) :

        input_batch = self.generate_input_batch(self.batch_size)
        logits_batch, policy_batch, value_batch = self.net.forward(input_batch)

        actions = torch.multinomial(policy_batch, num_samples=1)

        A = torch.squeeze(input=actions[:self.batch_size], dim=1)
        B = torch.squeeze(input=actions[self.batch_size:], dim=1)
        M = input_batch[:self.batch_size]

        player_one_rewards = M[torch.arange(self.batch_size), A][torch.arange(self.batch_size), B].view(self.batch_size, 1)

        rewards = torch.cat((player_one_rewards, -player_one_rewards), dim=0)
        
        advantages = (rewards - value_batch).flatten()

        baseline_loss = torch.sum(advantages**2) / 2

        cross_entropy = F.nll_loss(input=F.log_softmax(logits_batch, dim=1), target=actions.flatten(), reduction="none")
        policy_loss = torch.sum(cross_entropy * advantages.detach())

        loss = baseline_loss + policy_loss
        return loss

    def print (self, input_batch) :
        logits_batch, policy_batch, value_batch = self.net.forward(input_batch)

        
        # print("M:")
        # print(input_batch)
        # print("Policy Batch:")
        print(policy_batch)


if __name__ == '__main__':

    #plot shit

    triangle = np.array([[0, 1], [1, 0], [0, 0],[0, 1]])
    triangle_x = triangle[:, 0]
    triangle_y = triangle[:, 1]
    
    plt.plot(triangle_x, triangle_y)
    
    p_x = []
    p_y = []
    x = RNaD()

    RPS = torch.tensor([[0,-1,1],[1,0,-1],[-1,1,0]], dtype=torch.float).unsqueeze(dim=0)

    optimizer = torch.optim.SGD(x.net.parameters(), lr=.001)
    
    total_batches_exp = 10
    steps_exp = 5
    for steps in range(2**steps_exp):
        for _ in range(2**(total_batches_exp-steps_exp)):
            optimizer.zero_grad()
            loss = x.compute_loss_from_PG_step()

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits_batch, policy_batch, value_batch = x.net.forward(RPS)
            p_x.append(policy_batch[0, 0])
            p_y.append(policy_batch[0, 1])

        
    plt.plot(p_x, p_y)
    plt.show()
        


    