import time

import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from make_env import make_env
import matplotlib.pyplot as plt

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    scenario = 'Multi_UGV_Scenario'
    env = make_env(scenario, True)
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 100
    N_GAMES = 150000
    MAX_STEPS = 50
    total_steps = 0
    # task_completion = 0

    ind = []
    time_steps = []
    list_task_completion = []
    list_ag_col = []
    list_ob_col = []
    list_un_col = []
    score_history = []
    avg_score_history = []
    evaluate = False
    best_score = float('-inf')

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False] * n_agents
        episode_step = 0
        out_ag_col =  out_ob_col = out_un_col = out_reached_tar = 0
        #while not any(done):
        while (not all(done)) and episode_step < MAX_STEPS:
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            #print('Benchmark Data of Episode Step ', episode_step + 1)
            #print(info['n'])

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            #if episode_step >= MAX_STEPS:
            #    done = [True] * n_agents


            if out_reached_tar < n_agents:
                for j in range(n_agents):
                    done[j] = False
            else:
                for j in range(n_agents):
                    done[j] = True

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1
            # env.render()


            '''for a in env.agents:
                re, ag_col, ob_col, un_col, min_dis, reached_tar = env._get_info(a)
                out_reached_tar = reached_tar
                out_ag_col += ag_col
                out_ob_col += ob_col'''

            for j in range(n_agents):
                out_ag_col += (info['n'][j][1])
                out_ob_col += info['n'][j][2]
                out_un_col += info['n'][j][3]
                if info['n'][j][5] == n_agents:
                    out_reached_tar = n_agents

            '''print('Benchmark Data of Episode Step ', episode_step)
            print('Number of agent collisions: ', out_ag_col)
            print('Number of obstacle collisions: ', out_ob_col)
            print('Number of rough terrain collisions: ', out_un_col)
            print('Number of targets reached: ', out_reached_tar)
            print('Done List:', done)'''

        out_ag_col /= 2
        if out_reached_tar == n_agents:
            #task_completion += 1
            list_task_completion.append(1)
        else:
            list_task_completion.append(0)

        list_ag_col.append(int(out_ag_col))
        list_ob_col.append(out_ob_col)
        list_un_col.append(out_un_col)
        time_steps.append(episode_step)
        # tcr.append((task_completion / (i + 1)) * 100)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        ind.append(i + 1)

        save_location = 'tmp/run1/data_ep_'
        if(i % 1000 == 0):
            file1 = open(save_location + str(i) + '.txt', 'w')
            file1.writelines(str(list_task_completion))
            file1.writelines(str(list_ag_col))
            file1.writelines(str(list_ob_col))
            file1.writelines(str(list_un_col))
            file1.writelines(str(time_steps))
            file1.writelines(str(score_history))
            file1.writelines(str(avg_score_history))
            file1.close()

        '''print('Benchmark Data of Episode ', i)
        print(list_ag_col)
        print(list_ob_col)
        print(tcr)
        print(score_history)
        print(avg_score_history)'''

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                print('In episode ', i + 1, 'Best score reached: ', best_score)
        if (i + 1) % PRINT_INTERVAL == 0 and i > 0:
            print('In Episode', i + 1, 'average score {:.1f}'.format(avg_score))

    # Plotting

    '''plt.plot(ind, avg_score_history)
    plt.title('Average Score')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Score')
    plt.show()

    plt.plot(ind, list_ag_col)
    plt.title('Agent-Agent Collisions')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Agent-Agent Collisions')
    plt.show()

    plt.plot(ind, list_ob_col)
    plt.title('Agent-Obstacle Collisions')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Agent-Obstacle Collisions')
    plt.show()

    plt.plot(ind, list_ob_col)
    plt.title('Agent-Rough Terrain Collisions')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Agent-Rough Terrain Collisions')
    plt.show()

    plt.plot(ind, list_task_completion)
    plt.title('Task-Completion Rate')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Completed Tasks')
    plt.show()

    plt.plot(ind, score_history)
    plt.title('Score')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Score')
    plt.show()'''
