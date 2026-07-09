from collections import deque
import torch
from torch import nn
import numpy as np
import os

def get_activation(act_type: str):
    if act_type == 'LeakyRelu':
        return nn.LeakyReLU()
    elif act_type == 'Relu':
        return nn.ReLU()
    elif act_type == 'PRelu':
        return nn.PReLU()
    else:
        return nn.Identity()
    
# state_dim：状态输入维度；action_dim：动作输出维度；
# dueling：是否启用 Dueling 架构（将 Q 值分解为状态价值 V(s) 和优势函数 A(s,a)）。
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, activation: str = 'LeakyRelu',
                 hidden_layers: int = 2, dueling = False, scale = 1.):
        super(QNetwork, self).__init__()
        # 输入层：将状态维度映射到隐藏层维度
        self.in_layer = nn.Linear(state_dim, hidden_dim)#当输入状态向量（维度为 state_dim）传入时，通过线性变换 y = Wx + b 得到隐藏层向量（维度为 hidden_dim），完成维度映射。
         # 激活函数：根据参数选择（如 LeakyReLU、ReLU 等）
        self.act = get_activation(activation)
        self.dueling = dueling
        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)# 输出状态价值V(s)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)# 输出每个动作的优势值A(s, a)
        else:
            self.out_layer = nn.Linear(hidden_dim, action_dim)

        self.scale = scale

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        """
        ​​中间隐藏层​​：

            创建 hidden_layers个全连接层组成的列表

            每个层都是输入和输出维度均为 hidden_dim的线性层
        """
        self.mid_acts = nn.ModuleList([get_activation(activation) for _ in range(hidden_layers)])
        """
           为每个中间层创建一个对应的激活函数实例
        """

    def forward(self, observation):
            """
            定义网络如何将输入 observation(状态）转换为输出 Q 值

            observation形状:(batch_size, state_dim)或 (state_dim,):当输入是单个样本时,PyTorch会自动将其视为(1, state_dim)的批量
            """
            x = self.in_layer(observation)
            x = self.act(x)

            for mid_layer, mid_act in zip(self.mid_layers, self.mid_acts):
                x = mid_layer(x)
                x = mid_act(x)

            if self.dueling:
                value = self.value_stream(x)## (batch_size, hidden_dim) → (batch_size, 1)
                advantages = self.advantage_stream(x)# (batch_size, hidden_dim) → (batch_size, action_dim)
                x = value + (advantages - advantages.mean(dim=1, keepdim=True)) # → (batch_size, action_dim)

                """
                计算状态价值 V(s):(batch_size, 1)

                    计算动作优势 A(s,a):(batch_size, action_dim)

                    组合公式:Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]

                    mean(dim=1, keepdim=True)：对每个样本的所有动作优势值求平均
                
                """
            else:
                x = self.out_layer(x)

            if self.scale > 1:
                x *= self.scale
            return x


class DDQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)#实时更新，用于选动作 & 学习
        self.target_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)#延迟更新，用于计算目标 Q 值，减少训练震荡（DDQN 特性之一）
        self.target_net.load_state_dict(self.online_net.state_dict())#初始化目标网络参数为在线网络参数
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)#创建经验回放缓冲区（双端队列）用于存储经验 (s, mark, a, r, s’, done)
        self.batch_size = batch_size
        self.gamma = gamma#折扣因子 γ
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func#可选的“打乱函数”，用于对 (state, action) 做额外的随机变换
        self.repeat = repeat#每次调用 update 时，重复更新几次（相当于训练步数/经验利用率）

    def update(self, experiences):
        self.replay_buffer.extend(experiences)
        """
        # 假设 state 包含卫星队列状态、邻居负载等特征,mark=0表示无动作掩码
        [
            last_state,  # 丢失前的状态
            current_state[-1],  # 当前状态的最后一个元素（作为标记）
            2,  # 假设上一动作是“转发到邻居2”
            -5.2,  # 丢失的惩罚奖励
            current_state,  # 丢失后的状态
            1  # 任务结束
        ]
         self.propagator.experiences.append([last_state, last_state[-1], last_action, reward, current_state, done])
         存储强化学习的经验组，每个经验包含 [上一状态, 标记, 上一动作, 奖励, 当前状态, 是否结束]，用于智能体训练。
        """

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            """
            一次 update() 可以训练多步,repeat 控制重复次数。
            这样做的目的是提高每次采样的利用率。
            """
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]
            """
            从经验池里 随机采样 batch_size 条经验。
            replace=False 表示不重复采样。
            """

            state, mark, action, reward, next_state, done = zip(*batch)
            #	state-(batch_size, state_dim)
            #	next_state-(batch_size, state_dim)
            """
            解压后的 state 是一个包含 1024 个元素 的元组；
            元组中的每个元素是 65 维的 1D 向量（如 [0.3, 0.6, 0, ..., 1, 0])
                        state = (
                [0.2, 0.8, ..., 1, 0],  # 第1个经验的 last_state(65维1D向量)
                [0.1, 0.9, ..., 0, 1]   # 第2个经验的 last_state(65维1D向量)
            )
            """

            """
            state:当前状态
            mark:标记（决定使用 q_mask 还是所有动作，下面会用到）
            action:采取的动作
            reward:得到的奖励
            next_state:下一个状态
            done:是否结束(1=终止,0=继续）
            """
            state, action = self.shuffle((state, action))#调用 shuffle 方法，如果有传入 shuffle_func，就对 (state, action) 进行打乱或数据增强
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)
            curr_q = self.online_net(state)#输入状态，得到所有动作的 Q 值 (batch_size, action_dim)。
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()#只取出每个样本实际执行的动作对应的 Q 值（因为 online_net 会输出所有动作的 Q 值，我们只需要选中 a_t）
            next_q = self.online_net(next_state) #bug ---online_net->target_net
            next_q_1 = next_q[:, :self.q_mask].max(dim=1)[0]
            next_q_2 = next_q.max(dim=1)[0]
            next_q = mark * next_q_1 + (1 - mark) * next_q_2
            """
            用在线网络估计下一个状态的 Q 值
            next_q_1:只考虑前 q_mask 个动作
            next_q_2:考虑所有动作
            next_q = mark * next_q_1 + (1 - mark) * next_q_2:
            mark=1 时，选择 next_q_1
            mark=0 时，选择 next_q_2
             这说明 环境可能有两种不同的动作空间/约束条件，用 mark 来区分。

             这里使用的是 online_net 计算 next_q,而不是 target_net。
             这实际上更接近 普通 DQN,而不是 严格意义上的 Double DQN。
            （严格的 DDQN:选动作用 online_net,评估 Q 值用 target_net)
             """
            
            """
                    # ---------------------- Double DQN 的关键 ----------------------
            # 1. 动作选择：用 online_net 选择 next_state 下的动作
            next_q_online = self.online_net(next_state)
            best_next_actions = next_q_online.argmax(dim=1)

            # 2. 动作评估：用 target_net 计算 Q 值
            next_q_target = self.target_net(next_state)
            next_q = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            也就是说,Double DQN 用两个网络分工合作：
            online_net 用来决定 “选哪个动作”
            target_net 用来算 “这个动作的 Q 值是多少”
            # --------------------------------------------------------------

            """
            expected_q = reward + (1 - done) * self.gamma * next_q#y=r+γ⋅amax​Q(s′,a)
            """
            根据 Bellman 方程 计算目标 Q 值
            如果 done=1,说明游戏结束 → 没有未来奖励
            """
            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()#更新 online_net 的参数

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        print("Target network updated")

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path))
            self.target_net.load_state_dict(torch.load(file_path))

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences
        
class DQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_q = self.online_net(state)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            next_q = self.online_net(next_state).max(dim=1)[0]

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(file_path))

    def target_update(self):
        pass

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences


class PPO_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 1e-4, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling, scale= 1e2).to(device)
        self.critic_net = QNetwork(state_dim, hidden_dim, 1, activation, hidden_layers, dueling).to(device)

        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer_actor = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat

        self.eps_clip=0.1
        self.max_grad_norm = 0.5

    def update(self, experiences):

        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done = zip(*batch)
            action, old_log_prob = [a[0] for a in action], [a[1] for a in action]
            state, action = self.shuffle((state, action))
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            old_log_prob = torch.tensor(old_log_prob, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            with torch.no_grad():
                next_state = self.critic_net(next_state).squeeze()

            action_prob = self.online_net(state)

            mask = torch.ones_like(action_prob)
            mask[:, -1] = 0
            action_prob_1 = action_prob.masked_fill(mask == 0, float('-inf'))
            action_prob_1 = torch.nn.functional.softmax(action_prob_1, dim=-1)
            dist_1 = torch.distributions.Categorical(action_prob_1)

            action_prob = torch.nn.functional.softmax(action_prob, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action_log_prob = dist.log_prob(action)
            action_log_prob_1 = dist_1.log_prob(action)
            action_log_prob = action_log_prob_1 * mark + action_log_prob * (1-mark)

            state_value = self.critic_net(state).squeeze()

            advantages = reward + self.gamma * next_state * (1 - done) - state_value.detach()
            ratios = torch.exp(action_log_prob - old_log_prob.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()

            critic_loss = nn.functional.mse_loss(state_value, reward + self.gamma * next_state * (1 - done))
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        # torch.save(self.online_net.state_dict(), file_path + '/actor.pth')
        # torch.save(self.critic_net.state_dict(), file_path + '/critic.pth')
        # 使用os.path.join拼接路径，自动处理分隔符
        torch.save(self.online_net.state_dict(), os.path.join(file_path, 'actor.pth'))
        torch.save(self.critic_net.state_dict(), os.path.join(file_path, 'critic.pth'))

    def load_model(self,file_path):
        if file_path:
            # self.online_net.load_state_dict(torch.load(file_path + '/actor.pth', map_location=self.device))
            # self.critic_net.load_state_dict(torch.load(file_path + '/critic.pth', map_location=self.device))
            # 同样使用os.path.join拼接路径
            actor_path = os.path.join(file_path, 'actor.pth')
            critic_path = os.path.join(file_path, 'critic.pth')
            # 检查文件是否存在
            if not os.path.exists(actor_path):
                raise FileNotFoundError(f"模型文件不存在：{actor_path}")
            if not os.path.exists(critic_path):
                raise FileNotFoundError(f"模型文件不存在：{critic_path}")
            # 加载模型
            self.online_net.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences

def shuffle_neighbors(neighbor_states, other_states,action):
    parts = np.array_split(neighbor_states, 4)
    indices = np.random.permutation(4)
    new_state = np.concatenate([parts[idx] for idx in indices])
    if action < 4:
        new_action = int(np.where(indices == action)[0])
    else:
        new_action = action
    return np.concatenate([new_state, other_states]), new_action


class ShuffleEx:
    def __init__(self, shuffle_mask):
        self.shuffle_mask = shuffle_mask

    def shuffle(self, state, action):
        return shuffle_neighbors(state[:self.shuffle_mask], state[self.shuffle_mask:], action)



def cal_agent_dim(neighbors_dim: int, edges_dim: int, distance_dim: int, mission_dim: int, current_dim: int,
                  action_dim: int):
    return neighbors_dim + edges_dim + distance_dim + mission_dim + current_dim, action_dim, -(
            mission_dim + current_dim)
