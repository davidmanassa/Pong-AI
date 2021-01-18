"""

  Aqui treinamos um agente com gradientes de politica estocásticos.

"""

import numpy as np
import pickle as pickle
import gym

from gym import wrappers

# Parametros
H = 200 # Numero de neurónios da camada oculta

gamma = 0.99 # Fator de desconto para a recompensa

batch_size = 2 # numero de episódios até realizar um RMSprop
learning_rate = 1e-3 # learning rate usado no RMSprop
decay_rate = 0.99 # decay factor para o RMSProp

# flags de configuração
resume = True # inicializa o treino apartir do ultimo checkpoint (do ficheiro save.p)
render = True # interface

# Inicializaçao do modelo
D = 75 * 80 # dimensão do input: 75x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
  results = np.recfromcsv('results.csv').tolist()
else:
  results = []
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # Inicialização - A forma será H x D
  model['W2'] = np.random.randn(H) / np.sqrt(H) # forma H

# Matrizes identicas hás presentes no model, porêm preenchidas por zeros.
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # buffer para atualização dos gradientes
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # memória rmsprop

# Função sigmoid
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # Reduz o intervalo para [0, 1]

# Pré-processamento da imagem
def prepro(I):
  # Reduz uma imagem 210x160x3 uint8 em vetor de 6000 floats (75x80) 1D 
  I = I[35:185] # Recorte: removemos 35px do inicio e 25px do fim para remover partes redundantes, como quando a bola passa a raquete (paddle)
  I = I[::2,::2,0] # Reduzimos a imagem pelo fator de 2
  I[I == 144] = 0 # Apagamos o fundo (background type 1)
  I[I == 109] = 0 # Apagamos o fundo (background type 2)
  I[I != 0] = 1 # Tudo o resto (paddles, bola) é colocado a 1. Isto torna a imagem em escala de cinzentos eficientemente
  return I.astype(np.float).ravel() # ravel torna a matriz em um vetor de 1D

# Função de descontos
def discount_rewards(r):
  """
    r -> array 1D de floats de recompensas

    gera um array de descontos, desde a função mais recente para a mais antiga
    ações mais recentes irão ter um peso maior

  """
  discounted_r = np.zeros_like(r)
  running_add = 0 
  for t in reversed(range(0, r.size)):
    # quando alguem falha, reseta a soma, devido ao jogo ser por "rondas": fronteira
    # -> quando maior for uma "ronda", menor o desconto (dos ultimos passos)
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

# forward pass
def policy_forward(x):
  """ 
    Implementação manual do forward
  """
  # Cálculo das activações dos neurónios da camada oculta
  h = np.dot(model['W1'], x) # (H x D) . (D x 1) = (H x 1) (200 x 1)
  # ReLU não linear: limite em 0
  h[h<0] = 0
  # Probabilidade logarítmica de ir para cima
  logp = np.dot(model['W2'], h) # Retorna um decimal (1 x H) . (H x 1) = 1 (scalar)
  # Função sigmoid (probabilidade de ir para cima entre 0 e 1)
  p = sigmoid(logp)  # Coloca o retorno em um limite de [0, 1]
  return p, h # Retorna a probabilidade de tomar a ação 2 (UP), e a camada oculta 

# backward pass
def policy_backward(ep_h, ep_x, ep_lop_p):
  """ 
  implementação manual do backward pass
  
  ep_h é um array intermédio dos hidden states 
  ep_x é um array intermédio das imagens pré-processadas do jogo
  ep_log_p é um array intermédio das propriedades logaritmicas já descontadas

  Alimentamos a rede neuronal (para um episódio completo), e as suas probabilidades logaritmicas

  """
  # Ativação dos neurónios através da transposta das camadas ocultas e as suas probabilides logaritmicas
  dW2 = np.dot(ep_h.T, ep_lop_p).ravel()
  dh = np.outer(ep_lop_p, model['W2'])
  dh[ep_h <= 0] = 0 # backpro PReLU
  dW1 = np.dot(dh.T, ep_x)
  return {'W1':dW1, 'W2':dW2}

# Criamos o ambiente
env = gym.make("Pong-v0")

observation = env.reset()
prev_x = None # usamos para calcular o frame de diferença (motion)
game_xs, game_hs, game_log_p, game_rewards = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
  if render: env.render()

  # pre processamento da imagem
  cur_x = prepro(observation)
  # tomamos a diferença nos pixeis de input, visto que contem informações mais interessantes, como o Motion
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # encaminhamos a rede de politicas e seguimos uma ação da probabilidade de retorno
  aprob, h = policy_forward(x)
  
  # Aqui escolhemos um numero aleatório, para tomar uma ação aleatória.
  # Se o numero for menor que a probabilidade de subir da rede neurnal dando a imagem, vamos para baixo. Isto introduz a 'exploração' do agente.
  action = 2 if np.random.uniform() < aprob else 3 # 2 é para cima, 3 é para baixo, 0 é para manter
  
  # gravamos vários intermédios precisos para retropropagação
  game_xs.append(x) # observação
  game_hs.append(h) # estado oculto
  y = 1 if action == 2 else 0 # a "fake label" - this is the label that we're passing to the neural network to fake labels for supervised learning. It's fake because it is generated algorithmically, and not based on a ground truth, as is typically the case for Supervised learning

  # regista e encoraja a ação que será tomada a ser tomada
  game_log_p.append(y - aprob)

  # realiza ação no ambiente e obtem novas medidas
  observation, reward, done, info = env.step(action)
  """
  observation -> frames do jogo
  reward -> 0 enquanto nada acontecer, +1 se o oponente falhar a bola, -1 se nós falharmos a bola
  done -> quando um jogo acaba
  """

  reward_sum += reward

  # regista as recompensas 
  game_rewards.append(reward)

  if done: # Fim de um jogo/episódio
    episode_number += 1

    # empilha todos os inputs, estados escondidos, gradientes das ações, e recompensas para este episódio
    ep_x = np.vstack(game_xs)
    ep_h = np.vstack(game_hs)
    ep_lop_p = np.vstack(game_log_p)
    ep_rewards = np.vstack(game_rewards)
    game_xs, game_hs, game_log_p, game_rewards = [],[],[],[] # repõem os arrays

    # calcula a recompensa descontada 
    discounted_epr = discount_rewards(ep_rewards)
    # padroniza as recompensas para uma normal de unidades (ajuda a controlar a variancia do estimador do gradiente)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    ep_lop_p *= discounted_epr # modulamos o gradiente com vantagem (politica dos gradientes)
    grad = policy_backward(ep_h, ep_x, ep_lop_p)
    for k in model: grad_buffer[k] += grad[k] # soma dos gradientes

    # RMSprop a cada batch_size episódios 
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset ao buffer
        
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ("reset ao ambiente. recompensa total: %f" % (reward_sum))
    results.append(reward_sum)
    if episode_number % 100 == 0:
      pickle.dump(model, open('save.p', 'wb'))
      np.savetxt('results.csv', np.array(results), delimiter=',')
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # recompensa diferente de 0
    print ('ep %d: round finished, reward: %f' % (episode_number, reward))
