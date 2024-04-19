import numpy as np
from gymnasium.utils import EzPickle

from multiagent.mpe._mpe_utils.core import Agent, Landmark, World
from multiagent.mpe._mpe_utils.scenario import BaseScenario
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from multiagent.utils.conversions import parallel_wrapper_fn

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_prey=2,
        num_predators=4,
        num_obstacles=2,
        max_cycles=25,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_prey=num_prey,
            num_predators=num_predators,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_prey, num_predators, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
        )
        self.metadata["name"] = "predator_prey"
    def get_all_benchmark_data(self):
        # Assuming the scenario class has a method to calculate benchmark data
        return self.scenario.get_benchmark_data_for_all_predators(self.world)

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def make_world(self, num_prey=2, num_predators=4, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_prey_agents = num_prey
        num_predators = num_predators
        num_agents = num_predators + num_prey_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.predator = True if i < num_predators else False
            base_name = "predator" if agent.predator else "prey"
            base_index = i if i < num_predators else i - num_predators
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.predator else 0.05
            agent.accel = None
            agent.max_speed = 0.6 if agent.predator else 0.8
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.15
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.predator
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_vel = np.zeros(world.dim_p)

        world.landmarks[0].state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
        world.landmarks[1].state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
        # non-overlapping obstacles
        while (abs(world.landmarks[0].state.p_pos[0] - world.landmarks[1].state.p_pos[0])) ** 2 + (abs(world.landmarks[0].state.p_pos[1] - world.landmarks[1].state.p_pos[1])) ** 2 < (landmark.size*2) ** 2:
            world.landmarks[1].state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.predator:
            collisions = 0
            for a in self.prey_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def get_benchmark_data_for_all_predators(self, world):
        collision_num = 0
        for agent in world.agents:
            if agent.predator:
                collision_num += self.benchmark_data(agent, world)
        return collision_num
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not predators
    def prey_agents(self, world):
        return [agent for agent in world.agents if not agent.predator]

    # return all predators
    def predators(self, world):
        return [agent for agent in world.agents if agent.predator]

    def reward(self, agent, world):
        main_reward = (
            self.predator_reward(agent, world) #, shape=shape
            if agent.predator
            else self.agent_reward(agent, world)
        )
        return main_reward

    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by predators
    #     rew = 0
    #     shape = True
    #     predators = self.predators(world)
    #     if (
    #         shape
    #     ):  # reward can optionally be shaped (increased reward for increased distance from predators)
    #         for adv in predators:
    #             rew += 0.1 * np.sqrt(
    #                 np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
    #             )
    #     if agent.collide:
    #         for a in predators:
    #             if self.is_collision(a, agent):
    #                 rew -= 10

        
    #     # agents are penalized for exiting the screen, so that they can be caught by the predators
    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 10
    #         return min(np.exp(2 * x - 2), 10)

    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         rew -= bound(x)
        
    #     return rew
    
    def agent_reward(self, agent, world, shape=True, closer=False):
        def obstacle_avoidance_reward(agent, obstacles, avoidance_factor=0.1):
            # Reward agent based on distance to the closest obstacle
            min_distance = min(np.sqrt(np.sum(np.square(agent.state.p_pos - obs.state.p_pos))) for obs in obstacles)
            # Reward increases as distance to obstacle increases
            return avoidance_factor * min_distance

        # def exploration_bonus(agent, visited_positions, bonus=0.02):
        #     # Check if new position and reward for new locations
        #     if tuple(agent.state.p_pos) not in visited_positions:
        #         visited_positions.add(tuple(agent.state.p_pos))
        #         return bonus
        #     return 0
        
        def collaborative_reward(agent, other_agents, collaboration_radius=1.0, bonus=0.1, penalty=-0.1, closer=False):
            # Reward for staying close to other agents
            if closer:
                count_nearby = sum(1 for other in other_agents if np.linalg.norm(agent.state.p_pos - other.state.p_pos) < collaboration_radius)
                return bonus * count_nearby
            else: 
                # Reward for staying apart from other agents
                count_too_close = sum(1 for other in other_agents if np.linalg.norm(agent.state.p_pos - other.state.p_pos) < collaboration_radius)
                return penalty * count_too_close
        
        def incremental_distance_reward(agent, predators, factor=0.1):
            # Reward prey based on increasing distance to the closest predator
            min_distance = min(np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) for adv in predators)
            return factor * min_distance
        
        rew = 0
        obstacles = world.landmarks
        predators = self.predators(world)
        # last_position = agent.state.p_pos  # Assuming you keep track of last position

        # Apply reward functions
        if shape: 
            rew += obstacle_avoidance_reward(agent, obstacles)
            rew += incremental_distance_reward(agent, predators)
            # rew += exploration_bonus(agent, world.visited_positions)
            rew += collaborative_reward(agent, self.prey_agents(world), closer=closer)

                

        # Existing collision and boundary checks
        if agent.collide:
            for obs in obstacles:
                if self.is_collision(obs, agent):
                    rew -= 10
                    
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def predator_reward(self, agent, world):
        # predators are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.prey_agents(world)
        predators = self.predators(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in predators:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in predators:
                    if self.is_collision(ag, adv):
                        rew += 10
        
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.predator:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )
    #predator:[self_vel(2), self_pos(2), landmark_relative_pos(2*2), other_agents_relative_pos(2*3), prey_agent_vel(2)]
    #prey:[self_vel(2), self_pos(2), landmark_relative_pos(2*2), other_agents_relative_pos(2*3)]

    def caculate_boundary(self, world):
        for agent in world.agents:
            if agent.state.p_pos[0] > 1.0 or agent.state.p_pos[0] < -1.0:
                return False
            else:
                return True