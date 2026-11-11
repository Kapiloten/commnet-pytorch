import random
import torch
import torch.nn as nn
from torch.distributions import Categorical

nb_parties = 500
nb_episodes = 2000
N = 500
m = 5
nb_leviers = 5
leviers = [i for i in range(1, nb_leviers+1)]


def get_liste_agents():
    return random.sample([i for i in range(N)], m)

def choose_actions(liste_agents):
    actions = {}
    for i in liste_agents:
        actions[i] = random.choice(leviers)
    return actions

def calculate_score(actions) :
    nb_leviers_diff = torch.unique(actions).numel()
    return nb_leviers_diff/nb_leviers

# def run_experiments(nb_parties) :
#     scores = []
#     for i in range(nb_parties):
#         liste_agents = get_liste_agents()
#         actions = choose_actions(liste_agents)
#         score = calculate_score(actions)
#         scores.append(score)
#     return sum(scores)/len(scores)

def prepare_observations(liste_agents) :
    return torch.tensor(liste_agents, dtype=torch.long)

class AgentEncoder(nn.Module) :
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(N, 128)

    def forward(self, x) :
        return self.embed(x)

class AgentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche = nn.Linear(128, 5)

    def forward(self, x):
        return torch.softmax(self.couche(x), dim=1) # quand on aura (5, 5) on veut que les lignes qui sont les agents représentent des proba donc il faut sommer chaque ligne pour retrouver 1 donc il faut varier les colonnes donc il faut mettre dim=1

def select_actions(proba):
    distributions = Categorical(probs=proba)
    actions = distributions.sample()
    log_prob = distributions.log_prob(actions)
    return actions, log_prob

class IndependentController(nn.Module) :
    def __init__(self):
        super().__init__()

        self.Agentenc = AgentEncoder()
        self.Agentdec = AgentDecoder()

    def forward(self, x) :
        x = self.Agentenc(x)
        x = self.Agentdec(x)
        return x

def run_episode(model):
    liste_agents = get_liste_agents()
    liste_agents = prepare_observations(liste_agents)   
    proba_actions = model(liste_agents)
    actions, log_proba = select_actions(proba_actions)
    indices_leviers = actions + 1
    dict_actions = dict(zip(liste_agents.tolist(), indices_leviers.tolist()))
    score = calculate_score(actions)
    loss = - score * log_proba.sum()
    return score, loss


def train(model, opt, nb_episodes): 
    recomp = []
    for i in range(1, nb_episodes+1) :
        score, loss = run_episode(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        recomp.append(score)

        if i % 100 == 0 :
            moyenne = sum(recomp[-100:])/100
            print(f"Episode {i} | Récompense moyenne : {moyenne}")

    return recomp

Ind = IndependentController()
opt = torch.optim.Adam(Ind.parameters(), lr=0.001)
train(Ind, opt, nb_episodes)