#(c) Copywrite 2020 Aaron Krumins

# set async_mode to 'threading', 'eventlet', 'gevent' or 'gevent_uwsgi' to
# force a mode else, the best mode is selected automatically from what's
# installed
async_mode = None

#imports for communication between unreal and gymwrapper
import time
from flask import Flask, render_template
import logging
import logging.handlers
from engineio.async_drivers import eventlet
import socketio
from eventlet.support.dns import dnssec, e164, hash, namedict, tsigkeyring, update, version, zone
import json
import random
import numpy as np
import operator
import sys
import os
from gym import spaces
from random import randint

from threading import Timer


import ast
import math

#imports for open AI gym and a compatible machine library (stable baselines, neuroevolution etc)
import gym
import tensorflow as tf
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.deepq.policies import LnMlpPolicy as DqnLnMlpPolicy
from stable_baselines.deepq.policies import CnnPolicy as DqnCnnPolicy	
from stable_baselines.deepq.policies import LnCnnPolicy as DqnLnCnnPolicy	

from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy as SacLnMlpPolicy
from stable_baselines.sac.policies import CnnPolicy as SacCnnPolicy
from stable_baselines.sac.policies import LnCnnPolicy as SacLnCnnPolicy


from stable_baselines.td3.policies import MlpPolicy as Td3MlpPolicy
from stable_baselines.td3.policies import LnMlpPolicy as Td3LnMlpPolicy
from stable_baselines.td3.policies import CnnPolicy as Td3CnnPolicy
from stable_baselines.td3.policies import LnCnnPolicy as Td3LnCnnPolicy



from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN, PPO2, A2C, ACKTR, ACER, SAC, TD3 
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy

# Set up a specific logger with our desired output level
#logging.disable(sys.maxsize)

sio = socketio.Server(logger=True, async_mode = 'eventlet')
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)


thread = None


observations = "NaN"
UEreward = "0"
UEdone = False
maxactions = 0
obsflag = 0
inf = math.inf
actionspace = "Nan"
observationspace = "Nan"
results = os.getenv('APPDATA')       


# if getattr(sys, 'frozen', False):
    # application_path = os.path.dirname(sys.executable)
# elif __file__:
    # application_path = os.path.dirname(__file__)
#directory = "\\MindMaker"



def check_obs(self):
    global obsflag
    if (obsflag == 1):
        # observations recieved from UE, continue training
        obsflag = 0
    else:
        # observations not recieved yet, check again in a half second
        sio.sleep(.06)
        #check_obs(self)
        
   


class UnrealEnvWrap(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a env wrapper that recieves any environmental variables from UE and shapes into a format for OpenAI Gym 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}

  
  def __init__(self, ):
    super(UnrealEnvWrap, self).__init__()
    global maxactions
    global conactionspace
    global disactionspace
    #global minaction
    #global maxaction
    global actionspace
    global observationspace
    #print (minaction)
    #print (maxaction)
    print (conactionspace)
    

    print (actionspace)
    if conactionspace == True:
        print("continous action space")
        actionspace = "spaces.Box(" + actionspace + ")"
        observationspace = "spaces.Box(" + observationspace + ")"
        #self.action_space = spaces.Box(ast.literal_eval(actionspace))  
        self.action_space = eval(actionspace) 
        #self.agent_pos = randint(0, maxactions)   
        self.agent_pos = randint(0, 100)   
        #low = np.array([-2,0,-100])
        #high = np.array([2, 100, 100])
        #self.observation_space = spaces.Box(low=np.array([-2,0,-100]), high=np.array([2,100,100]),dtype=np.float32)    
        self.observation_space = eval(observationspace)
    elif disactionspace == True:        
        # Initialize the agent with a random action
        print("discrete action space")
        actionspace = int(actionspace)
        self.agent_pos = randint(0, actionspace)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = actionspace
        self.action_space = spaces.Discrete(n_actions)
        observationspace = "spaces.Box(" + observationspace + ")"
        # The observation will be all environment variables from UE that agent is tracking
        n_actionsforarray = n_actions - 1
        #low = np.array([0,0])
        #high = np.array([n_actionsforarray, n_actionsforarray])
        #self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = eval(observationspace)
    else:
        logmessages = "No action space type selected"
        sio.emit('messages', logmessages)

  
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent with a random action
    self.observation_space = [0]
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.observation_space])

    
    
 #sending actions to UE and recieving observations in response to those actions  
  def step(self, action):
    global observations
    global UEreward
    global UEdone
    global obsflag
    obsflag = 0

    #send actions to UE as they are chosen by the RL algorityhm
    straction = str(action)
    print("action:",straction)
    sio.emit('recaction', straction)
    #After sending action, we enter a pause loop until we recieve a response from UE with the observations
    for i in range(10000):
        if obsflag == 1:
            obsflag = 0
            break
        else:
            sio.sleep(.06)
    #load the observations recieved from UE4
    arrayobs = ast.literal_eval(observations)
    self.observation_space = arrayobs
    print(arrayobs)
    done = bool(UEdone)
    reward = float(UEreward)
    print("reward", reward)
    print(UEdone)
    if done == True:
        print("Im rrestarting now how fun")
        reward = 0
    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return np.array([self.observation_space]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()


  def close(self):
    os._exit(1)
    pass




@sio.event
def disconnect_request(sid):
    sio.disconnect(sid)
    os._exit(1)

@sio.event
def connect(sid, environ):
    print ("Connected To Unreal Engine")


@sio.event
def disconnect(sid):
    print('Disconnected From Unreal Engine, Exiting MindMaker')
    os._exit(1)
    

    
@sio.on('launchmindmaker')
def recieve(sid, data):

    global UEdone
    global reward
    global maxactions
    global conactionspace
    global disactionspace
    #global minaction
    #global maxaction
    global actionspace
    global observationspace
    jsonInput = json.loads(data);
    actionspace = jsonInput['actionspace'];
    observationspace = jsonInput['observationspace'];
    #minaction = jsonInput['minaction'];
    #maxaction = jsonInput['maxaction'];
    #maxactions = jsonInput['maxactions'];
    trainepisodes = jsonInput['trainepisodes']
    evalepisodes = jsonInput['evalepisodes']
    loadmodel = jsonInput['loadmodel']
    savemodel = jsonInput['savemodel']
    modelname = jsonInput['modelname']
    algselected = jsonInput['algselected']
    usecustomparams = jsonInput['customparams']
    a2cparams = jsonInput['a2cparams']
    acerparams = jsonInput['acerparams']
    acktrparams = jsonInput['acktrparams']
    dqnparams = jsonInput['dqnparams']
    # ddpgparams = jsonInput['ddpgparams']
    # ppo1params = jsonInput['ppo1params']
    ppo2params = jsonInput['ppo2params']
    sacparams = jsonInput['sacparams']
    td3params = jsonInput['td3params']
    # trpoparams = jsonInput['trpoparams']
    conactionspace = jsonInput['conactionspace']
    disactionspace = jsonInput['disactionspace']
    totalepisodes = trainepisodes + evalepisodes 
    UEdone = jsonInput['done']
    env = UnrealEnvWrap()
    # wrap it
    env = make_vec_env(lambda: env, n_envs=1)
    print("save model value:", savemodel)
    print("load model value:", loadmodel)
    
    path = results + "\\" + modelname
    
    
    if loadmodel == 'true':
    # Load the trained agent
        if algselected == 'DQN':
            model = DQN.load(path)
        elif algselected == 'A2C':
            model = A2C.load(path)
        elif algselected == 'ACER':
            model = ACER.load(path)
        elif algselected == 'ACKTR':
            model = ACKTR.load(path)
        elif algselected == 'DDPG (Requires Microsoft OpenMPI)':
            from stable_baselines.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines.ddpg.policies import LnMlpPolicy as DdpgLnMlpPolicy
            from stable_baselines.ddpg.policies import CnnPolicy as DdpgCnnPolicy
            from stable_baselines.ddpg.policies import LnCnnPolicy as DdpgLnCnnPolicy
            from stable_baselines import DDPG
            print("DDPG requires Microsoft Open MPI be installed on your system")
            model = DDPG.load(path)
        elif algselected == 'PPO1 (Requires Microsoft OpenMPI)':
            from stable_baselines import PPO1
            model = PPO1.load(path)
        elif algselected == 'PPO2':
            model = PPO2.load(path)
        elif algselected == 'SAC':
            model = SAC.load(path)
        elif algselected == 'TD3':
            model = TD3.load(path)
        elif algselected == 'TRPO (Requires Microsoft OpenMPI)':
            from stable_baselines import TRPO
            model = TRPO.load(path)
        
        print("Loading the Trained Agent")
        logmessages = "Loading the Trained Agent"
        sio.emit('messages', logmessages)
        obs = env.reset()
        intaction = 0
        #Begin strategic behvaior
        for step in range(evalepisodes):
          action, _ = model.predict(obs, deterministic=True)
          intaction = action[0]
          print("Action: ", intaction)
          obs, reward, done, info = env.step(action)
          print('obs=', obs, 'reward=', reward, 'done=', done)
    

    else:
                # Train the agent with different algorityhms from stable baselines
        
        #model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./DQN_newobservations/")
        print("alg selected:", algselected)
        print("use custom:", usecustomparams)
        
        if (algselected == 'DQN') and (usecustomparams == 'true'):
            gammaval = dqnparams["gamma"]
            policyval = dqnparams["policy"]
            act_funcval = dqnparams["act func"]
            learning_rateval = dqnparams["learning rate"]
            verboseval = dqnparams["verbose"]
            tensorboard_logval = dqnparams["tensorboard log"]
            _init_setup_modelval = dqnparams["init setup model"]
            full_tensorboard_logval = dqnparams["full tensorboard log"]
            seedval = dqnparams["seed"]
            n_cpu_tf_sessval = dqnparams["n cpu tf sess"]
            layersval = dqnparams["layers"]
            buffer_sizeval = dqnparams["buffer size"]
            exploration_fractionval = dqnparams["exploration fraction"]
            exploration_final_epsval = dqnparams["exploration final eps"]
            exploration_initial_epsval = dqnparams["exploration initial eps"]
            batch_sizeval = dqnparams["batch size"]
            train_freqval = dqnparams["train freq"]
            double_qval = dqnparams["double q"]
            learning_startsval = dqnparams["learning starts"]
            prioritized_replayval = dqnparams["prioritized replay"]
            target_network_update_freqval = dqnparams["target network update freq"]
            prioritized_replay_alphaval = dqnparams["prioritized replay alpha"]
            prioritized_replay_beta0val = dqnparams["prioritized replay beta0"]
            prioritized_replay_beta_itersval = dqnparams["prioritized replay beta iters"]
            prioritized_replay_epsval = dqnparams["prioritized replay eps"]
            param_noiseval = dqnparams["param noise"]

            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
                print("tanh")
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
                print("relu")
            elif act_funcval == 'tf.nn.leaky_relu' :
                print("leaky_relu")
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))            
            print(policyval)
            policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = DQN(eval(policyval), env, gamma = gammaval, learning_rate = learning_rateval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = ast.literal_eval(buffer_sizeval), exploration_fraction = exploration_fractionval, exploration_final_eps = exploration_final_epsval, exploration_initial_eps = exploration_initial_epsval, batch_size = batch_sizeval, train_freq = train_freqval, double_q = double_qval, learning_starts = learning_startsval, target_network_update_freq = target_network_update_freqval, prioritized_replay = prioritized_replayval, prioritized_replay_alpha = prioritized_replay_alphaval, prioritized_replay_beta0 = prioritized_replay_beta0val, prioritized_replay_beta_iters = ast.literal_eval(prioritized_replay_beta_itersval), prioritized_replay_eps = prioritized_replay_epsval, param_noise = ast.literal_eval(param_noiseval) )
            
            #model = DQN(DqnMlpPolicy, env,  gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000, target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None )
            
            print("Custom DQN training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
            
        elif algselected == 'DQN':
            model = DQN(DqnMlpPolicy, env, verbose=1)
            print("DQN training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif (algselected == 'A2C') and (usecustomparams == 'true') :
            policyval = a2cparams["policy"]
            act_funcval = a2cparams["act func"]
            gammaval = a2cparams["gamma"]
            n_stepsval = a2cparams["n steps"]
            vf_coefval = a2cparams["vf coef"]
            ent_coefval = a2cparams["ent coef"]
            
            learning_rateval = a2cparams["learning rate"]
            alphaval = a2cparams["alpha"]
            epsilonval = a2cparams["epsilon"]
            lr_scheduleval = a2cparams["lr schedule"]
            verboseval = a2cparams["verbose"]
            tensorboard_logval = a2cparams["tensorboard log"]
            _init_setup_modelval = a2cparams["init setup model"]
            
            seedval = a2cparams["seed"]
            
            
            
            
            
            full_tensorboard_logval = a2cparams["full tensorboard log"]
            n_cpu_tf_sessval = a2cparams["n cpu tf sess"]
            max_grad_normval = a2cparams['max grad norm']
            
            network_archval = a2cparams["network arch"]
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, net_arch = ast.literal_eval(network_archval))
            print(policyval)
            model = A2C(policyval, env, gamma = gammaval, n_steps=n_stepsval, ent_coef= ent_coefval, max_grad_norm = max_grad_normval, learning_rate = learning_rateval, alpha = alphaval, epsilon = epsilonval, lr_schedule = lr_scheduleval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval))

            print("Custom A2C training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'A2C':
            model = A2C(MlpPolicy, env, verbose=1)
            print("A2C training in process...")
            model.learn(total_timesteps=trainepisodes)
            
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                
        elif (algselected == 'ACER') and (usecustomparams == 'true'):
            gammaval = acerparams["gamma"]
            policyval = acerparams["policy"]
            act_funcval = acerparams["act func"]
            n_stepsval = acerparams["n steps"]
            ent_coefval = acerparams["ent coef"]
            max_grad_normval = acerparams["max grad norm"]
            learning_rateval = acerparams["learning rate"]
            alphaval = acerparams["alpha"]
            lr_scheduleval = acerparams["lr schedule"]
            verboseval = acerparams["verbose"]
            num_procsval = acerparams["num procs"]
            tensorboard_logval = acerparams["tensorboard log"]
            _init_setup_modelval = acerparams["init setup model"]
            full_tensorboard_logval = acerparams["full tensorboard log"]
            seedval = acerparams["seed"]
            n_cpu_tf_sessval = acerparams["n cpu tf sess"]
            network_archval = acerparams["network arch"]
            q_coefval = acerparams["q coef"]
            rprop_alphaval = acerparams["rprop alpha"]
            rprop_epsilonval = acerparams["rprop epsilon"]
            buffer_sizeval = acerparams["buffer size"]
            replay_ratioval = acerparams["replay ratio"]
            replay_startval = acerparams["replay start"]
            correction_termval = acerparams["correction term"]
            trust_regionval = acerparams["trust region"]

            deltaval = acerparams["delta"]
            
            print("polic val:", policyval)
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, net_arch = ast.literal_eval(network_archval))
            
            
            model = ACER(policyval, env, gamma = gammaval, n_steps=n_stepsval, ent_coef= ent_coefval, max_grad_norm = max_grad_normval, learning_rate = learning_rateval, alpha = alphaval, lr_schedule = lr_scheduleval, verbose = verboseval, num_procs = ast.literal_eval(num_procsval), tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), q_coef = q_coefval, rprop_alpha = rprop_alphaval, rprop_epsilon = rprop_epsilonval, buffer_size = buffer_sizeval, replay_ratio = replay_ratioval, replay_start = replay_startval, correction_term = float(correction_termval),trust_region = trust_regionval, delta = deltaval )

            print("Custom ACER training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'ACER':
            model = ACER(MlpPolicy, env, verbose=1)
            print("ACER training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif (algselected == 'ACKTR') and (usecustomparams == 'true'):
            gammaval = acktrparams["gamma"]
            policyval = acktrparams["policy"]
            act_funcval = acktrparams["act func"]
            n_stepsval = acktrparams["n steps"]
            ent_coefval = acktrparams["ent coef"]
            max_grad_normval = acktrparams["max grad norm"]
            learning_rateval = acktrparams["learning rate"]
            lr_scheduleval = acktrparams["lr schedule"]
            verboseval = acktrparams["verbose"]
            tensorboard_logval = acktrparams["tensorboard log"]
            _init_setup_modelval = acktrparams["init setup model"]
            full_tensorboard_logval = acktrparams["full tensorboard log"]
            seedval = acktrparams["seed"]
            n_cpu_tf_sessval = acktrparams["n cpu tf sess"]
            network_archval = acktrparams["network arch"]          
            nprocsval = acktrparams["nprocs"]
            vf_coefval = acktrparams["vf coef"]
            vf_fisher_coefval = acktrparams["vf fisher coef"]
            kfac_clipval = acktrparams["kfac clip"]
            async_eigen_decompval = acktrparams["async eigen decomp"]
            kfac_updateval = acktrparams["kfac update"]
            gae_lambdaval = acktrparams["gae lambda"]
            #policy_kwargsval = dict(net_arch = ast.literal_eval(network_archval))
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, net_arch = ast.literal_eval(network_archval))            

            model = ACKTR(policyval, env, gamma = gammaval, n_steps=n_stepsval, ent_coef= ent_coefval, max_grad_norm = max_grad_normval, learning_rate = learning_rateval, lr_schedule = lr_scheduleval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), nprocs = ast.literal_eval(nprocsval), vf_coef = vf_coefval, vf_fisher_coef = vf_fisher_coefval, kfac_clip = kfac_clipval, async_eigen_decomp = async_eigen_decompval, kfac_update = kfac_updateval, gae_lambda = ast.literal_eval(gae_lambdaval) )

            print("Custom ACKTR training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        
        elif algselected == 'ACKTR':
            model = ACKTR(MlpPolicy, env, verbose=1)
            print("ACKTR training in process...")
            model.learn(total_timesteps=trainepisodes)
        elif (algselected == 'DDPG (Requires Microsoft OpenMPI)') and (usecustomparams == 'true'):
            from stable_baselines.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines.ddpg.policies import LnMlpPolicy as DdpgLnMlpPolicy
            from stable_baselines.ddpg.policies import CnnPolicy as DdpgCnnPolicy
            from stable_baselines.ddpg.policies import LnCnnPolicy as DdpgLnCnnPolicy
            from stable_baselines import DDPG
            print("DDPG requires Microsoft Open MPI be installed on your system")
            gammaval = ddpgparams["gamma"]
            policyval = ddpgparams["policy"]
            act_funcval = ddpgparams["act_func"]
            #learning_rateval = ddpgparams["learning_rate"]
            verboseval = ddpgparams["verbose"]
            tensorboard_logval = ddpgparams["tensorboard_log"]
            _init_setup_modelval = ddpgparams["_init_setup_model"]
            full_tensorboard_logval = ddpgparams["full_tensorboard_log"]
            seedval = ddpgparams["seed"]
            n_cpu_tf_sessval = ddpgparams["n_cpu_tf_sess"]
            layersval = dqnparams["layers"]         
            buffer_sizeval = ddpgparams["buffer_size"]
            eval_envval = ddpgparams["eval_env"]
            nb_train_stepsval = ddpgparams["nb_train_steps"]
            nb_rollout_stepsval = ddpgparams["nb_rollout_steps"]
            #batch_sizeval = ddpgparams["batch_size"]
            nb_eval_stepsval = ddpgparams["nb_eval_steps"]
            action_noiseval = ddpgparams["action_noise"]
            param_noise_adaption_intervalval = ddpgparams["param_noise_adaption_interval"]
            tauval = ddpgparams["tau"]
            normalize_returnsval = ddpgparams["normalize_returns"]
            critic_l2_regval = ddpgparams["critic_l2_reg"]
            enable_popartval = ddpgparams["enable_popart"]
            normalize_observationsval = ddpgparams["normalize_observations"]
            observation_rangeval = ddpgparams["observation_range"]
            #return_rangeval = ddpgparams["return_range"]
            param_noiseval = ddpgparams["param_noise"]
            actor_lrval = ddpgparams["actor_lr"]    
            critic_lrval = ddpgparams["critic_lr"]    
            clip_normval = ddpgparams["clip_norm"]    
            reward_scaleval = ddpgparams["reward_scale"]
            renderval = ddpgparams["render"] 
            render_evalval = ddpgparams["render_eval"]       
            memory_limitval = ddpgparams["memory_limit"]   
            memory_policyval = ddpgparams["memory_policy"]   
            random_explorationval = ddpgparams["random_exploration"]   
            
            random_explorationval = float(random_explorationval)
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))         
                
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = DDPG(eval(policyval), env, gamma = gammaval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = ast.literal_eval(buffer_sizeval), eval_env = ast.literal_eval(eval_envval), nb_train_steps = nb_train_stepsval, nb_rollout_steps = nb_rollout_stepsval, nb_eval_steps = nb_eval_stepsval, tau = tauval, param_noise_adaption_interval = param_noise_adaption_intervalval, normalize_returns = normalize_returnsval, critic_l2_reg = ast.literal_eval(critic_l2_regval), enable_popart = enable_popartval, normalize_observations = normalize_observationsval, observation_range = ast.literal_eval(observation_rangeval), return_range = (-inf, inf), param_noise = ast.literal_eval(param_noiseval), actor_lr = actor_lrval, critic_lr = critic_lrval, clip_norm = ast.literal_eval(clip_normval), reward_scale = reward_scaleval, render = ast.literal_eval(renderval), render_eval = ast.literal_eval(render_evalval), memory_limit = ast.literal_eval(memory_limitval), memory_policy = ast.literal_eval(memory_policyval), random_exploration = random_explorationval, action_noise = ast.literal_eval(action_noiseval) )
            
            #model = DDPG(DdpgMlpPolicy,env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50, nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None, normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False, observation_range=(-5.0, 5.0), critic_l2_reg=0.0, return_range=(-inf, inf), actor_lr=0.0001, critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1) 

            print("Custom DDPG training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("DDPG training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'DDPG (Requires Microsoft OpenMPI)':
            from stable_baselines.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines.ddpg.policies import LnMlpPolicy as DdpgLnMlpPolicy
            from stable_baselines.ddpg.policies import CnnPolicy as DdpgCnnPolicy
            from stable_baselines.ddpg.policies import LnCnnPolicy as DdpgLnCnnPolicy
            from stable_baselines import DDPG
            print("DDPG requires Microsoft Open MPI be installed on your system")
            # the noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            param_noise = None
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            model = DDPG(DdpgMlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
            print("DDPG training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("DDPG training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        
        elif (algselected == 'PPO1 (Requires Microsoft OpenMPI)') and (usecustomparams == 'true'):
            from stable_baselines.ppo1 import PPO1
            from stable_baselines import PPO1
            gammaval = ppo1params["gamma"]
            act_funcval = ppo1params["act_func"]
            policyval = ppo1params["policy"]


            timesteps_per_actorbatchval = ppo1params["timesteps_per_actorbatch"]

            verboseval = ppo1params["verbose"]
            tensorboard_logval = ppo1params["tensorboard_log"]
            _init_setup_modelval = ppo1params["_init_setup_model"]
            full_tensorboard_logval = ppo1params["full_tensorboard_log"]
            seedval = ppo1params["seed"]
            n_cpu_tf_sessval = ppo1params["n_cpu_tf_sess"]
            layersval = dqnparams["layers"]         
            
            clip_paramval = ppo1params["clip_param"]
            
            entcoeffval = ppo1params["entcoeff"]
            optim_epochsval = ppo1params["optim_epochs"]
            optim_stepsizeval = ppo1params["optim_stepsize"]
            optim_batchsizeval = ppo1params["optim_batchsize"]
            lamval = ppo1params["lam"]
            adam_epsilonval = ppo1params["adam_epsilon"]
            
            scheduleval = ppo1params["schedule"]
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = PPO1(policyval, env, gamma = gammaval, timesteps_per_actorbatch = timesteps_per_actorbatchval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), clip_param = clip_paramval, entcoeff = entcoeffval, optim_epochs = optim_epochsval, optim_stepsize = optim_stepsizeval, optim_batchsize = optim_batchsizeval, lam = lamval, schedule = scheduleval, adam_epsilon = adam_epsilonval )
            

            print("Custom PPO1 training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("PPO1 training complete")
            if savemodel == 'true':
                # Save the agent
                path = results + "\\" + modelname
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
               
        
        elif algselected == 'PPO1 (Requires Microsoft OpenMPI)':
            from stable_baselines.ppo1 import PPO1
            from stable_baselines import PPO1
            model = PPO1(MlpPolicy, env, verbose=1)
            print("PPO1 training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("PPO1 training complete")  
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif (algselected == 'PPO2') and (usecustomparams == 'true'):
            gammaval = ppo2params["gamma"]
            policyval = ppo2params["policy"]
            act_funcval = ppo2params["act func"]
            n_stepsval = ppo2params["n steps"]
            verboseval = ppo2params["verbose"]
            tensorboard_logval = ppo2params["tensorboard log"]
            _init_setup_modelval = ppo2params["init setup model"]
            full_tensorboard_logval = ppo2params["full tensorboard log"]
            seedval = ppo2params["seed"]
            n_cpu_tf_sessval = ppo2params["n cpu tf sess"]
            layersval = ppo2params["layers"]         
            ent_coefval = ppo2params["ent coef"]
            learning_rateval = ppo2params["learning rate"]
            vf_coefval = ppo2params["vf coef"]
            nminibatchesval = ppo2params["nminibatches"]
            noptepochsval = ppo2params["noptepochs"]
            lamval = ppo2params["lam"]
            cliprangeval = ppo2params["cliprange"]
  
            cliprange_vfval = ppo2params["cliprange vf"]
  
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = PPO2(policyval, env, gamma = gammaval, n_steps = n_stepsval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = ast.literal_eval(full_tensorboard_logval), seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), ent_coef = ent_coefval, learning_rate = learning_rateval, vf_coef = vf_coefval,  nminibatches =  nminibatchesval, noptepochs = noptepochsval, lam = lamval, cliprange_vf = ast.literal_eval(cliprange_vfval), cliprange = cliprangeval )

            #model = PPO2(MlpPolicy, env, gamma = gammaval, n_steps=128, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
            print("Custom PPO2 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
        elif algselected == 'PPO2':
            model = PPO2(MlpPolicy, env, verbose=1)
            print("PPO2 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
        elif (algselected == 'SAC') and (usecustomparams == 'true'):
            gammaval = sacparams["gamma"]
            policyval = sacparams["policy"]
            act_funcval = sacparams["act func"]
            learning_rateval = sacparams["learning rate"]
            verboseval = sacparams["verbose"]
            tensorboard_logval = sacparams["tensorboard log"]
            _init_setup_modelval = sacparams["init setup model"]
            full_tensorboard_logval = sacparams["full tensorboard log"]
            seedval = sacparams["seed"]
            n_cpu_tf_sessval = sacparams["n cpu tf sess"]
            layersval = sacparams["layers"]                   
            buffer_sizeval = sacparams["buffer size"]   
            batch_sizeval = sacparams["batch size"]
            train_freqval = sacparams["train freq"]
            learning_startsval = sacparams["learning starts"]  
            tauval = sacparams["tau"]
            ent_coefval = sacparams["ent coef"]
            target_update_intervalval = sacparams["target update interval"]
            gradient_stepsval = sacparams["gradient steps"]           
            target_entropyval = sacparams["target entropy"]
            action_noiseval = sacparams["action noise"]
            random_explorationval = sacparams["random exploration"]
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))
            model = SAC(eval(policyval), env, gamma = gammaval, learning_rate = learning_rateval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = buffer_sizeval, tau = tauval, ent_coef = ent_coefval, target_update_interval = target_update_intervalval, batch_size = batch_sizeval, train_freq = train_freqval, gradient_steps = gradient_stepsval, learning_starts = learning_startsval,  action_noise = ast.literal_eval(action_noiseval), random_exploration = random_explorationval, target_entropy = target_entropyval)
            
            
            #model = SAC(eval(policyval), env, gamma = 0.99, learning_rate = 0.0003, verbose = 0, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = False, seed = None, n_cpu_tf_sess = None, buffer_size = buffer_sizeval, tau = 0.005, ent_coef = 'auto', target_update_interval = 1, batch_size = 64, train_freq = 1, gradient_steps = 1, learning_starts = learning_startsval,  action_noise = None, random_exploration = 0.0, target_entropy = 'auto')
            
            #model = SAC(eval(policyval), env, gamma=0.99, learning_rate=0.0003, buffer_size=50000, learning_starts=100, train_freq=1, batch_size=64, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1, target_entropy='auto', action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

            print("Custom SAC training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        elif algselected == 'SAC':
            model = SAC(SacMlpPolicy, env, verbose=1)
            print("SAC training in process...") 
            model.learn(total_timesteps=trainepisodes, log_interval=10)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)                
                
        elif (algselected == 'TD3') and (usecustomparams == 'true'):
            gammaval = td3params["gamma"]
            policyval = td3params["policy"]
            act_funcval = td3params["act func"]
            learning_rateval = td3params["learning rate"]
            verboseval = td3params["verbose"]
            tensorboard_logval = td3params["tensorboard log"]
            _init_setup_modelval = td3params["init setup model"]
            full_tensorboard_logval = td3params["full tensorboard log"]
            seedval = td3params["seed"]
            n_cpu_tf_sessval = td3params["n cpu tf sess"]
            layersval = td3params["layers"]                   
            buffer_sizeval = td3params["buffer size"]   
            batch_sizeval = td3params["batch size"]
            
            learning_startsval = td3params["learning starts"]  
            tauval = td3params["tau"]
            
            policy_delayval = td3params["policy delay"]

            action_noiseval = td3params["action noise"]
            random_explorationval = td3params["random exploration"]

            train_freqval = td3params["train freq"]
            target_noise_clipval = td3params["target noise clip"]
            gradient_stepsval = td3params["gradient steps"]           
            
            target_policy_noiseval = td3params["target policy noise"]
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            
            
            model = TD3(eval(policyval), env, gamma = gammaval, learning_rate = learning_rateval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = buffer_sizeval, tau = tauval, target_noise_clip = target_noise_clipval, policy_delay = policy_delayval, batch_size = batch_sizeval, train_freq = train_freqval, gradient_steps = gradient_stepsval, learning_starts = learning_startsval,  action_noise = ast.literal_eval(action_noiseval), random_exploration = random_explorationval, target_policy_noise = target_policy_noiseval  )

            print("Custom TD3 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")            
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)        
        
        elif algselected == 'TD3':
            # The noise objects for TD3
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = TD3(Td3MlpPolicy, env, action_noise=action_noise, verbose=1)
            print("TD3 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        elif (algselected == 'TRPO (Requires Microsoft OpenMPI)') and (usecustomparams == 'true'):
            from stable_baselines import TRPO
            gammaval = trpoparams["gamma"]
            policyval = trpoparams["policy"]
            act_funcval = trpoparams["act_func"]
            timesteps_per_batchval = trpoparams["timesteps_per_batch"]
            verboseval = trpoparams["verbose"]
            tensorboard_logval = trpoparams["tensorboard_log"]
            _init_setup_modelval = trpoparams["_init_setup_model"]
            full_tensorboard_logval = trpoparams["full_tensorboard_log"]
            seedval = trpoparams["seed"]
            n_cpu_tf_sessval = trpoparams["n_cpu_tf_sess"]
            layersval = trpoparams["layers"]                   
            max_klval = trpoparams["max_kl"]   
            cg_itersval = trpoparams["cg_iters"]
            
            lamval = trpoparams["lam"]  
            entcoeffval = trpoparams["entcoeff"]
            
            cg_dampingval = trpoparams["cg_damping"]

            vf_stepsizeval = trpoparams["vf_stepsize"]
            vf_itersval = trpoparams["vf_iters"]

            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = TRPO(policyval, env, gamma = gammaval, timesteps_per_batch = timesteps_per_batchval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), max_kl = max_klval, entcoeff = entcoeffval, cg_damping = cg_dampingval, cg_iters = cg_itersval,  lam = lamval,  vf_stepsize = vf_stepsizeval, vf_iters = vf_itersval,  )

            print("Custom TRPO training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        elif algselected == 'TRPO (Requires Microsoft OpenMPI)':
            from stable_baselines import TRPO
            model = TRPO(MlpPolicy, env, verbose=1)
            print("TRPO training in process...") 
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        #model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./A2C_newobservations/")
        #model = A2C(MlpPolicy, env, verbose=1)
        else:
            print("No learning algorithm selected for training with")
            logmessages = "No learning algorithm selected for training with"
            sio.emit('messages', logmessages)
            sio.disconnect(sid)
    
        # Test the trained agent, (currently not needed, all testing occurs in Unreal itself)
        

        
        env.render(mode='console')
        #env.render()

        obs = env.reset()
        print("Training complete")
        logmessages = "Training complete"
        sio.emit('messages', logmessages)
        intaction = 0
        #Begin strategic behvaior
        evalcomplete = evalepisodes + 2
        print(evalcomplete)
        for step in range(evalcomplete):
                action, _  = model.predict(obs, deterministic=True)
                intaction = action[0]
                print("Action: ", intaction)
                obs, reward, done, info = env.step(action)
                print('obs=', obs, 'reward=', reward, 'done=', done)
                if step == evalepisodes:
                    print(step)
                    logmessages = "Evaluation Complete"
                    sio.emit('messages', logmessages)


                    
        
        
    sio.disconnect(sid)




#recieves observations and reward from Unreal Engine    
@sio.on('sendobs')
def sendobs(sid, obsdata):
    global obsflag
    global observations
    global UEreward
    global UEdone

    obsflag = 1
    # print("Mindmaker recieved observations and reward from UE, passing back an action")
    # jassi
    # don't see this change in the output
    print("Mindmaker recieved observations and reward from UE, passing back an action HELLO")
    # jassiend
    jsonInput = json.loads(obsdata);
    observations = jsonInput['observations']     
    UEreward = jsonInput['reward'];
    UEdone = jsonInput['done'];
    



#This sets up the server connection, with UE acting as the client in a socketIO relationship, will default to eventlet    
if __name__ == '__main__':
    # jassi
    # doesn't do a thing but generally, i don't see any changes i make in this file
    # original_stdout = sys.stdout # Save a reference to the original standard output
    # f = open('mindmakeroutput.txt', 'w')
    # sys.stdout = f # Change the standard output to the file we created.
    # jassiend

    if sio.async_mode == 'threading':
        # deploy with Werkzeug
        print("1 ran")
        app.run(threaded=True)
        
    elif sio.async_mode == 'eventlet':
        # deploy with eventlet
        import eventlet
        import eventlet.wsgi
        logging.disable(sys.maxsize)
        print("MindMaker running, waiting for Unreal Engine to connect")
        eventlet.wsgi.server(eventlet.listen(('', 3000)), app)
    elif sio.async_mode == 'gevent':
        # deploy with gevent
        from gevent import pywsgi
        try:
            from geventwebsocket.handler import WebSocketHandler
            websocket = True
            print("3 ran")
        except ImportError:
            websocket = False
        if websocket:
            pywsgi.WSGIServer(('', 3000), app, log = None,
                              handler_class=WebSocketHandler).serve_forever()
            print("4 ran")
            log = logging.getLogger('werkzeug')
            log.disabled = True
            app.logger.disabled = True
        else:
            pywsgi.WSGIServer(('', 3000), app).serve_forever()
            print("5 ran")
    elif sio.async_mode == 'gevent_uwsgi':
        print('Start the application through the uwsgi server. Example:')
        print('uwsgi --http :5000 --gevent 1000 --http-websockets --master '
              '--wsgi-file app.py --callable app')
    else:
        print('Unknown async_mode: ' + sio.async_mode)

    # jassi
    # sys.stdout = original_stdout # Reset the standard output to its original value
    # jassiend
   
    

