import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

size = 3
#global hyper parameter


################


class Net (torch.nn.Module):

    def __init__ (self):
        super().__init__()

        self.fc0 = nn.Linear(size**2, size**3)
        self.fc1 = nn.Linear(size**3, size)
        self.fc2 = nn.Linear(size**3, 1)

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
        self.validation_batch = torch.cat((self.validation_batch, -self.validation_batch.clone().detach().swapaxes(1,2)), dim=0)

        self.net = Net()
        self.net_checkpoints = {}

        self.gamma = .1
        self.total_NeuRD_steps = 2**10
        self.total_RNaD_steps = 2**10
    
    def reset_net (self):
        self.net = Net()

    def generate_input_batch (self, batch_size : int, matrix_entry_distribution=None) -> torch.tensor :
        #currenly bypasses generator and uses normal
        if matrix_entry_distribution is None:
            matrix_entry_distribution = torch.Generator() #fix so zero mean

        #player_one_batch = torch.rand(size=(batch_size, size, size), generator=matrix_entry_distribution)
        player_one_batch = torch.normal(0, 1, size=(batch_size, size, size))
        player_two_batch = player_one_batch.clone().detach()
        player_two_batch = -player_two_batch.swapaxes(1, 2) #zero sum
        input_batch = torch.cat((player_one_batch, player_two_batch), dim=0)
        return input_batch

    def exploitability_ (self, input_batch):
        batch_size = input_batch.shape[0] // 2
        with torch.no_grad():
            logits_batch, policy_batch, value_batch = self.net.forward(input_batch)

        M = input_batch[:batch_size]
        A = policy_batch[:batch_size].view(batch_size, 1, size)
        B = policy_batch[batch_size:].view(batch_size, size, 1)

        # u = torch.matmul(B, torch.matmul(M, A))
        X = torch.matmul(A, M)
        Y = torch.matmul(M, B)
        min_X = torch.min(X, dim=2)[0]
        max_Y = torch.max(Y, dim=1)[0]

        expl = max_Y - min_X

        return logits_batch, policy_batch, value_batch, expl

    def exploitability (self, input_batch, policy_batch):
        batch_size = input_batch.shape[0] // 2

        M = input_batch[:batch_size]
        A = policy_batch[:batch_size].view(batch_size, 1, size)
        B = policy_batch[batch_size:].view(batch_size, size, 1)

        # u = torch.matmul(B, torch.matmul(M, A))
        X = torch.matmul(A, M)
        Y = torch.matmul(M, B)
        min_X = torch.min(X, dim=2)[0]
        max_Y = torch.max(Y, dim=1)[0]

        expl = max_Y - min_X

        return expl

    def exploitability_demo_PG (self, total_batches_exp, steps_exp, lr=.01, update_rule='neurd'):
        #plot the mean exploitability on validation batch

        optimizer = torch.optim.SGD(x.net.parameters(), lr)
        
        mean_expls = []
        policy_batch_total = torch.zeros(self.validation_batch.shape[:2])
        policy_batch_checkpoints = []
        for steps in range(2**steps_exp):
            for _ in range(2**(total_batches_exp-steps_exp)):
                optimizer.zero_grad()


                loss = x.compute_loss_from_PG_step()

                loss.backward()
                optimizer.step()

            logits_batch, policy_batch, value_batch, expl = self.exploitability_(self.validation_batch)
            policy_batch_total += policy_batch
            time_averaged_policy = policy_batch_total / (steps + 1)

            expl = self.exploitability(self.validation_batch, time_averaged_policy)

            policy_batch_checkpoints.append(policy_batch)
            mean_expls.append(torch.mean(expl).item())
            # mean exploitability of mean (time averaged) strategies
        
        fig, ax = plt.subplots()
        ax.plot(range(2**steps_exp), mean_expls)
        ax.set_ylim([0, 4])

        plt.show() 
        print(mean_expls[-1])
        return   

    def exploitability_demo_NeuRD (self, total_batches_exp, steps_exp, lr=.01):
        #plot the mean exploitability on validation batch

        optimizer = torch.optim.SGD(x.net.parameters(), lr)
        
        mean_expls = []
        policy_batch_total = torch.zeros(self.validation_batch.shape[:2])
        policy_batch_checkpoints = []
        for steps in range(2**steps_exp):
            for _ in range(2**(total_batches_exp-steps_exp)):
                optimizer.zero_grad()
                loss = x.compute_loss_from_NeuRD_step()

                loss.backward()
                optimizer.step()

            logits_batch, policy_batch, value_batch, expl = self.exploitability_(self.validation_batch)
            policy_batch_total += policy_batch
            time_averaged_policy = policy_batch_total / (steps + 1)

            expl = self.exploitability(self.validation_batch, time_averaged_policy)

            policy_batch_checkpoints.append(policy_batch)
            mean_expls.append(torch.mean(expl).item())
            # mean exploitability of mean (time averaged) strategies
        
        fig, ax = plt.subplots()
        ax.plot(range(2**steps_exp), mean_expls)
        ax.set_ylim([0, 4])
        plt.show()    
        print(mean_expls[-1])

    def compute_loss_from_PG_step (self) :

        RPS = torch.tensor([[0,-.7,.8],[.6, 0,  -.9],[-.9, .9, 0]], dtype=torch.float).unsqueeze(dim=0)

        input_batch = torch.cat((RPS, -RPS.swapaxes(1,2)), dim=0)
        logits_batch, policy_batch, value_batch = self.net.forward(input_batch)

        actions = torch.multinomial(policy_batch, num_samples=1)

        A = torch.squeeze(input=actions[:self.batch_size], dim=1)
        B = torch.squeeze(input=actions[self.batch_size:], dim=1)
        M = input_batch[:self.batch_size]

        player_one_rewards = M[torch.arange(self.batch_size), A][torch.arange(self.batch_size), B].view(self.batch_size, 1)
        rewards = torch.cat((player_one_rewards, -player_one_rewards), dim=0)
        advantages = (rewards - value_batch).flatten()
        chosen_logit_batch = logits_batch[torch.arange(self.batch_size * 2), torch.squeeze(input=actions, dim=1)].flatten()
        chosen_policy_batch = policy_batch[torch.arange(self.batch_size * 2), torch.squeeze(input=actions, dim=1)].flatten()

        policy_loss = torch.mean(-chosen_logit_batch * advantages.detach() / chosen_policy_batch.detach())

        baseline_loss = torch.mean(advantages**2) / 2

        entropy_loss = torch.mean(policy_batch * F.log_softmax(logits_batch, dim=1))

        loss = baseline_loss + policy_loss + entropy_loss
        return loss

    def compute_loss_from_NeuRD_step (self) :

        input_batch = self.generate_input_batch(self.batch_size)
        logits_batch, policy_batch, value_batch = self.net.forward(input_batch)

        try:
            actions = torch.multinomial(policy_batch, num_samples=1)
        except:
            print(policy_batch)

        A = torch.squeeze(input=actions[:self.batch_size], dim=1)
        B = torch.squeeze(input=actions[self.batch_size:], dim=1)
        M = input_batch[:self.batch_size]

        player_one_rewards = M[torch.arange(self.batch_size), A][torch.arange(self.batch_size), B].view(self.batch_size, 1)
        rewards = torch.cat((player_one_rewards, -player_one_rewards), dim=0)
        advantages = (rewards - value_batch).flatten()
        chosen_logit_batch = logits_batch[torch.arange(self.batch_size * 2), torch.squeeze(input=actions, dim=1)].flatten()
        chosen_policy_batch = policy_batch[torch.arange(self.batch_size * 2), torch.squeeze(input=actions, dim=1)].flatten()

        policy_loss = torch.mean(-chosen_logit_batch * advantages.detach() / chosen_policy_batch.detach())

        baseline_loss = torch.mean(advantages**2) / 2

        entropy_loss = torch.mean(policy_batch * F.log_softmax(logits_batch, dim=1))

        loss = baseline_loss + policy_loss #+ entropy_loss
        return loss

    def RPS_animation (self, total_batches_exp, steps_exp):
        triangle = np.array([[0, 1], [1, 0], [0, 0],[0, 1]])
        triangle_x = triangle[:, 0]
        triangle_y = triangle[:, 1]

        p_x = []
        p_y = []


        RPS = torch.tensor([[0,-1,1],[1,0,-1],[-1,1,0]], dtype=torch.float).unsqueeze(dim=0)

        optimizer = torch.optim.SGD(x.net.parameters(), lr=.001)
        
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
        fig, ax = plt.subplots()

        # Convert to time average
        # p_x = [sum(p_x[:_]) / _ for _ in range(1, 2**steps_exp+1)]
        # p_y = [sum(p_y[:_]) / _ for _ in range(1, 2**steps_exp+1)]
        steps_text = ["Param update steps: {}".format((_+1) * 2**(total_batches_exp-steps_exp)) for _ in range(2**steps_exp)]

        def animate(i):
            length = 10
            # q_x = p_x[i-length: i]
            # q_y = p_y[i-length: i]

            ax.clear()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.text(.6, .6, steps_text[i])

            for _ in range(length):
                z = max(0, i+_-length)
                ax.plot(p_x[z:z+2], p_y[z:z+2], alpha=(_/length), color='black')
            
            # ax.plot(q_x, q_y, alpha=.2)
            
            ax.plot(triangle_x, triangle_y)
            ax.plot([1/3], [1/3], '--bo')

        print("Final mean", torch.mean(torch.tensor(p_x)), torch.mean(torch.tensor(p_y)))

        ani = animation.FuncAnimation(fig, animate, frames=2**steps_exp, interval=6, repeat=True)
        writergif = animation.PillowWriter(fps=30)
        ani.save("/home/user/Desktop/d.gif", writer=writergif)
        plt.show()    

 

if __name__ == '__main__':

    #plot shit
    x = RNaD()
    x.batch_size = 1
    x.RPS_animation(12, 9)
    # x.batch_size = 2**7
    # n = 10
    # x.exploitability_demo_NeuRD(n, n, .01)
    # x.reset_net()
    # x.exploitability_demo_PG(n, n, .01)


#     def print (self, input_batch) :
#         logits_batch, policy_batch, value_batch = self.net.forward(input_batch)
#         print(policy_batch)


# def RPS_animation_NeuRD (self, total_batches_exp, steps_exp):
#         triangle = np.array([[0, 1], [1, 0], [0, 0],[0, 1]])
#         triangle_x = triangle[:, 0]
#         triangle_y = triangle[:, 1]
    
#         p_x = []
#         p_y = []


#         RPS = torch.tensor([[0,-1,1],[1,0,-1],[-1,1,0]], dtype=torch.float).unsqueeze(dim=0)

#         optimizer = torch.optim.SGD(x.net.parameters(), lr=.01)
        
#         for steps in range(2**steps_exp):
#             for _ in range(2**(total_batches_exp-steps_exp)):
#                 optimizer.zero_grad()
#                 loss = x.compute_loss_from_PG_step()

#                 loss.backward()
#                 optimizer.step()

#             with torch.no_grad():
#                 logits_batch, policy_batch, value_batch = x.net.forward(RPS)
#                 p_x.append(policy_batch[0, 0])
#                 p_y.append(policy_batch[0, 1])
#         fig, ax = plt.subplots()

#         # Convert to time average
#         p_x = [sum(p_x[:_]) / _ for _ in range(1, 2**steps_exp+1)]
#         p_y = [sum(p_y[:_]) / _ for _ in range(1, 2**steps_exp+1)]
#         steps_text = ["Param update steps: {}".format((_+1) * 2**(total_batches_exp-steps_exp)) for _ in range(2**steps_exp)]

#         def animate(i):
#             length = 10
#             # q_x = p_x[i-length: i]
#             # q_y = p_y[i-length: i]

#             ax.clear()
#             ax.set_xlim([0, 1])
#             ax.set_ylim([0, 1])
#             ax.text(.6, .6, steps_text[i])

#             for _ in range(length):
#                 z = max(0, i+_-length)
#                 ax.plot(p_x[z:z+2], p_y[z:z+2], alpha=(_/length), color='black')
            
#             # ax.plot(q_x, q_y, alpha=.2)
            
#             ax.plot(triangle_x, triangle_y)
#             ax.plot([1/3], [1/3], '--bo')

#         print("Final mean", torch.mean(torch.tensor(p_x)), torch.mean(torch.tensor(p_y)))

#         ani = animation.FuncAnimation(fig, animate, frames=2**steps_exp, interval=6, repeat=True)
#         # writergif = animation.PillowWriter(fps=30)
#         # ani.save("/home/user/Desktop/a.gif", writer=writergif)
#         plt.show()  