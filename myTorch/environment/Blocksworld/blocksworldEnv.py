#!/usr/bin/env python
__author__ = "Prasanna"
__credits__ = ["Sarath", "Chinna"]

import pygame, math, sys
import os
from pygame.locals import *
import numpy as np
import time
from copy import deepcopy
import matplotlib.pyplot as plt

UNIT_DISTANCE = 20
UNIT_DISTANCE_X = 30
HEIGHT = 150
MAX_OBJECTS = 5
MAX_LOCATIONS = 10
OFFSET = 340
NO_OF_BLOCKS = 12
NUM_COLORS = 4
TABLE = 20

class Agent_Sprite(pygame.sprite.Sprite):
    MAX_FORWARD_SPEED = 0
    MAX_REVERSE_SPEED = 0
    ACCELERATION = 0
    TURN_SPEED = 0
    #np.random.seed(0)
    def __init__(self, image, position,index):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load(image)
        self.src_image = pygame.transform.scale(self.src_image, (30, 20))
        self.position = position
        self.speed = self.direction = 0
        self.holding = False
        self.location = None
        self.holding_object = None
        self.image = None
        self.index = index
    def update(self,action,agent, locations):
        # SIMULATION
        if action!= None:
            if action == 'pick' and self.holding == False:
                if self.location.objects.index(agent)>0:
                    self.position = (self.position[0],self.position[1]+UNIT_DISTANCE)
                    self.holding_object = self.location.objects[self.location.objects.index(agent)-1]
                    self.location.swap(agent, self.holding_object)
                    self.holding = True
                    self.holding_object.update(action,self.location,self.position[1])
            elif action == 'right':
                if locations.index(self.location)<MAX_LOCATIONS-1:
                    l1 = len(self.location.objects)
                    self.location.remove(agent)
                    locations[locations.index(self.location)] = self.location
                    locations[locations.index(self.location)+1].add(agent)
                    self.location = locations[locations.index(self.location)+1]


                    if self.holding == True:
                        self.position = (self.position[0]+UNIT_DISTANCE_X, self.position[1] - (len(self.location.objects)+1-l1)*UNIT_DISTANCE)
                        self.holding_object.update(action,self.location,self.position[1])
                        locations[locations.index(self.location)].add(self.holding_object)
                        locations[locations.index(self.location)-1].remove(self.holding_object)
                    else:
                        self.position = (self.position[0]+UNIT_DISTANCE_X, self.position[1] - (len(self.location.objects)-l1)*UNIT_DISTANCE)
                    self.location = locations[locations.index(self.location)]


            elif action == 'left':
                if locations.index(self.location)>0:
                    l1 = len(self.location.objects)
                    self.location.remove(agent)
                    locations[locations.index(self.location)] = self.location
                    locations[locations.index(self.location)-1].add(agent)
                    self.location = locations[locations.index(self.location)-1]

                    if self.holding == True:
                        self.position = (self.position[0]-UNIT_DISTANCE_X, self.position[1] - (len(self.location.objects)+1-l1)*UNIT_DISTANCE)
                        self.holding_object.update(action,self.location,self.position[1])
                        locations[locations.index(self.location)].add(self.holding_object)
                        locations[locations.index(self.location)+1].remove(self.holding_object)

                    else:
                        self.position = (self.position[0]-UNIT_DISTANCE_X,self.position[1] - (len(self.location.objects)-l1)*UNIT_DISTANCE)
                    self.location = locations[locations.index(self.location)]

            elif action == 'drop' and self.holding == True:
                if self.location.objects.index(agent)<MAX_OBJECTS-1:
                    self.location = self.location.swap(self.holding_object,agent)
                    locations[locations.index(self.location)] = self.location
                    self.position = (self.position[0],HEIGHT-(len(self.location.objects)-1)*UNIT_DISTANCE)
                    self.holding_object.update(action,self.location,self.position[1])
                    #self.location.add
                    self.holding = False
                    self.holding_object = None
                    #self.location.swap(self.holding_object,agent)
                    #self.position = (self.position[0],self.position[1]+UNIT_DISTANCE)

            self.image = pygame.transform.rotate(self.src_image, self.direction)
            self.rect = self.image.get_rect()
            self.rect.center = self.position
            return locations

class Block_Sprite(pygame.sprite.Sprite):
    MAX_FORWARD_SPEED = 0
    MAX_REVERSE_SPEED = 0
    ACCELERATION = 0
    TURN_SPEED = 0
    def __init__(self, image_dir, position,index,color, number = None):
        pygame.sprite.Sprite.__init__(self)
        self.number = number
        if self.number!=None:
            self.src_image1 = pygame.image.load(os.path.join(image_dir, str(number+1)+'.bmp'))
        self.src_image = pygame.image.load(os.path.join(image_dir, color +'.bmp'))
        m = 1
        if index == 'bar' or index == 'table':
            m=5
        self.src_image = pygame.transform.scale(self.src_image, (int(30/m), int(20*m)))
        if self.number!=None:
            self.src_image1 = pygame.transform.scale(self.src_image, (int(30/m), int(20*m)))
    #    self.src_image = pygame.transform.scale(self.src_image, (30, 20))
        self.position = position
        self.speed = self.direction = 0
        self.location = None
        self.previous_action = None
        self.index = index
        self.color = color
    def update(self, action,location, agent_y):
        # SIMULATION
        if location !=None:
            self.location = location
            if action == 'pick':
                self.position = (self.position[0],self.position[1]-UNIT_DISTANCE)
                self.previous_action = 'pick'
            elif action == 'right':
                self.position = (self.position[0]+UNIT_DISTANCE_X,agent_y-UNIT_DISTANCE)#-len(self.location.objects)*UNIT_DISTANCE)
                self.previous_action = 'move'
            elif action == 'left':
                self.position = (self.position[0]-UNIT_DISTANCE_X,agent_y-UNIT_DISTANCE)#-len(self.location.objects)*UNIT_DISTANCE)
                self.previous_action = 'move'
            elif action == 'drop':
                self.previous_action = None
                self.position = (self.position[0],HEIGHT - (len(self.location.objects)-2)*UNIT_DISTANCE)
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position
        if self.number!=None:
            self.image1 = pygame.transform.rotate(self.src_image1, self.direction)
            self.rect1 = self.image1.get_rect()
            self.rect1.center = self.position

class location:
    def __init__(self):
        self.objects = []
        self.top_object = None

    def remove(self,object_sel):
        self.objects.remove(object_sel)

    def swap(self, object1, object2):
        #swaps object positions
        #removes the two objects and adds object1 and object2 in that order
        self.remove(object1)
        self.remove(object2)
        self.objects.append(object1)
        self.objects.append(object2)
        return self

    def top_object(self):
        return self.objects[-1]
    def add(self,object1):
        self.objects.append(object1)

class all_locations:
    def __init__(self):
        self.location_order = []

    def add(self,loc):
        self.location_order.append(loc)

    def next_location(self,curr_loc):
        c_l = self.location_order.index(curr_loc)
        if c_l == len(self.location_order)-1:
            return None
        else:
            return self.location_order[c_l+1]

    def previous_location(self,curr_loc):
        c_l = self.location_order.index(curr_loc)
        if c_l == 0:
            return None
        else:
            return self.location_order[c_l-1]

class Environment:
    '''Initialize with mode. Mode can be \'colors\' or \'None\'. Where colors define a
    world with colored blocks and the goal is checked with matching the location with exact stacking of blocks of same color.
    None defines a world with all blocks of blue color and the goal condition is checked only by matching the
    number of blocks on target locations with current location.'''
    def __init__(self,mode, image_dir, target = None, problem =0, max_episode_len=50):
        self.mode = mode
        self.image_dir = image_dir
        self.actions = ['left','right','pick','drop']
        self.colors = ['red','blue','green','purple']
        self.target = target
        self.problem = problem
        self.observation_space = tuple([6,315,200])
        self.inpos = -100
        self.max_episode_len = max_episode_len
        self.curr_episode_len = 0 

    def reset(self,mode,target = None,problem =0, min_b = 1, max_b = 10):
        '''Resets the world and creates a new instance of the game with new target and state'''
        if target!=None:
            self.target = target
        self.mode = mode
        self.bar1 = Block_Sprite(self.image_dir, (TABLE-UNIT_DISTANCE_X,HEIGHT),'bar','bar')
        self.bar2 = Block_Sprite(self.image_dir, (OFFSET-10,HEIGHT - UNIT_DISTANCE),'bar','bar')
        self.tab = Block_Sprite(self.image_dir, (TABLE+UNIT_DISTANCE_X,HEIGHT+50),'table','tab')
        self.Blocks =[]
        self.T_Blocks = []
        self.problem =problem
        self.target_dict = {}
        self.selected_colors = []
        self.color_dict = {}
        assert min_b <= max_b
        #np.random.seed(2)
        NO_OF_BLOCKS = 1#np.random.randint(min_b,max_b)
        if self.target != None:
            D = open(self.target,'r').readlines()
            Locs = D[self.problem].split('#')
            assert len(Locs)-1 == MAX_LOCATIONS
            for i in range(len(Locs)-1):
                loc = Locs[i].split(':')
                if len(loc[1])>0:
                    self.selected_colors+=loc[1].split(',')
                    self.target_dict.update({loc[0]:loc[1].split(',')})
                else:
                    self.target_dict.update({loc[0]:[]})
            assert len(self.selected_colors) == NO_OF_BLOCKS
        for i in range(NO_OF_BLOCKS):
            if self.target == None:
                if self.mode == 'colors':
                    #np.random.seed(2)
                    r = np.random.randint(NUM_COLORS)
                else:
                    r = 1
                self.selected_colors+=[self.colors[r]]
                r = self.colors[r]
            else:
                r = self.selected_colors[i]
            self.Blocks+= [Block_Sprite(self.image_dir, (TABLE,HEIGHT),i,r)]
            self.color_dict.update({i:self.selected_colors[i]})
        self.Agent = Agent_Sprite(os.path.join(self.image_dir, 'agent_stand.bmp'),(TABLE,HEIGHT),'Agent')
        self.All_locations = [location() for i in range(MAX_LOCATIONS)]
        self.Target = [location() for i in range(MAX_LOCATIONS)]
        for i in range(NO_OF_BLOCKS):
            #r = np.random.randint(len(self.selected_colors))
            self.T_Blocks += [Block_Sprite(self.image_dir, (TABLE,HEIGHT),i,self.color_dict[i])]
            #self.selected_colors.remove(self.selected_colors[r])
        self.Blocks += [self.Agent]

        def update_table(Blocks,All_locations,include_offset):
            '''Updates the items on the table.'''
            rnd_c =0
            for b in Blocks:
                #np.random.seed(rnd_c)
                rnd_c+=1
                r = np.random.randint(20)
                flag = False
                while len(All_locations[(int((b.position[0]-TABLE)/UNIT_DISTANCE_X)+r)%MAX_LOCATIONS].objects)>=MAX_OBJECTS-2 or flag == False:
                    if flag ==True:
                        #np.random.seed(rnd_c+3)
                        r = np.random.randint(20)
                    flag = True
                    All_locations[(int((b.position[0]-TABLE)/UNIT_DISTANCE_X)+r)%MAX_LOCATIONS].add(b)
                    b.location = All_locations[(int((b.position[0]-TABLE)/UNIT_DISTANCE_X)+r)%MAX_LOCATIONS]
                    b.position = (OFFSET*include_offset+b.position[0]+(All_locations.index(b.location)%MAX_LOCATIONS)*UNIT_DISTANCE_X,b.position[1]-(All_locations[All_locations.index(b.location)].objects.index(b))*UNIT_DISTANCE)
            return Blocks
        self.Blocks = update_table(self.Blocks,self.All_locations,0)
        if self.target == None:
            self.T_Blocks = update_table(self.T_Blocks,self.Target,1)
        else:
            comp_ind = []
            for k,v in self.target_dict.items():
                v_c = deepcopy(v)
                v_c_dict = {}
                c = 0
                if len(v_c)>0:
                    for item in v_c:
                        if item not in v_c_dict:
                            v_c_dict.update({item:[c]})
                        else:
                            v_c_dict[item]+=[c]
                        c+=1
                    for b in self.T_Blocks:
                        if b.index not in comp_ind and b.color in v_c_dict:
                            loc = int(k)
                            if len(v_c_dict[b.color]) >0:
                                pos = v_c_dict[b.color][0]
                                upd = v_c_dict[b.color][1:]
                                v_c_dict.update({b.color:upd})
                                self.Target[loc-1].add(b)
                                b.location = self.Target[loc-1]
                                b.position = (OFFSET+b.position[0]+(loc-1)*UNIT_DISTANCE_X,b.position[1]-pos*UNIT_DISTANCE)
                                comp_ind+=[b.index]
        self.RenderInit()
        _, _, image = self.step(3)
        self.curr_episode_len = 0
        return image

    def RenderInit(self):
        '''Initializes the rendering objects.'''
        self.screen = pygame.display.set_mode((700,200),RESIZABLE)
        self.b_w = pygame.sprite.RenderPlain(self.Agent)
        self.bar_rend = pygame.sprite.RenderPlain(self.bar1)
        self.bar_rend1 = pygame.sprite.RenderPlain(self.bar2)
        self.tab_rend = pygame.sprite.RenderPlain(self.tab)
        self.blocks_render = [pygame.sprite.RenderPlain(b) for b in self.Blocks]
        self.t_blocks_render = [pygame.sprite.RenderPlain(b) for b in self.T_Blocks]
        self.blocks_render += [self.bar_rend1]

    def check(self):
        cur_st = self.All_locations
        target_ = self.Target
        flag = True
        in_pos = 0
        reward = 0
        for i in range(MAX_LOCATIONS):
            tar = []
            curr = []
            for b in target_[i].objects:
                if b.index!='Agent':
                    if self.mode == 'None':
                        tar+=[1]
                    elif self.mode == 'colors':
                        tar+=[b.color]
            for b in cur_st[i].objects:
                if b.index!='Agent':
                    if self.mode == 'None':
                        curr+=[1]
                    elif self.mode == 'colors':
                        curr+=[b.color]
            for k in range(min(len(curr),len(tar))):
                if curr[k] == tar[k]:
                    in_pos+=1
            if curr!=tar:
                flag = False
        if self.inpos==-100:
            self.inpos = in_pos
            reward = 0
        else:
            reward = in_pos - self.inpos
            self.inpos = in_pos
        if flag:
            reward = 10
        return (flag,reward)

    def step(self,action):
        ''' The main action step function in class Environment. Valid action arguments are  \'0-left,1-right,2-pick,3-drop\'
        this function returns a tuple [boolean, nparray, nparray] for [goal_achieved, current_state, target_state]'''
        # USER INPUT
        action =self.actions[action]
        self.screen.fill((255,255,255))
        self.All_locations = self.Agent.update(action,self.Agent,self.All_locations)
        for b in self.blocks_render:
            b.update(None,None,0)
            b.draw(self.screen)
        for b in self.t_blocks_render:
            b.update(None,None,0)
            b.draw(self.screen)
        self.b_w.draw(self.screen)
        for i in range(MAX_LOCATIONS):
            pygame.draw.line(self.screen, (0, 0, 0), (TABLE+(i)*UNIT_DISTANCE_X-15, HEIGHT+10), (TABLE+(i+1)*UNIT_DISTANCE_X-15, HEIGHT+10))
            pygame.draw.line(self.screen, (0, 0, 0), (OFFSET+TABLE+(i)*UNIT_DISTANCE_X-15, HEIGHT+10), (OFFSET+TABLE+(i+1)*UNIT_DISTANCE_X-15, HEIGHT+10))
        pygame.display.flip()
        screen_copy = self.screen.copy()
        rect = pygame.Rect(5, 0, 315, 200)
        sub1 = screen_copy.subsurface(rect)
        rect = pygame.Rect(345, 0 , 315, 200)
        sub2 = screen_copy.subsurface(rect)
        self.RenderInit()
        done, reward = self.check()
        image = np.transpose(np.concatenate([pygame.surfarray.pixels3d(sub1),pygame.surfarray.pixels3d(sub2)],axis=2))
        self.curr_episode_len += 1
        if self.curr_episode_len > self.max_episode_len:
            done = True
        return done, reward, image

    def random_step(self):
        '''A random agent that selects actions uniformly at random'''
        return self.step(np.random.randint(len(self.actions)))
