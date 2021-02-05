#!/usr/bin/python


import numpy as np
from pyrep import PyRep
import numpy as np

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

from matplotlib import cm
cols = cm.get_cmap('tab20c', 7)

class BlockSim:
    
    def __init__(self,move_tolerance=1e-3):
        
        self.pr = PyRep()
        self.pr.launch(headless=False)
        
        self.pr.start()
        
        self.move_tolerance = move_tolerance
        
        self.blocks = ['T','p','V','L','Z','b','a']
    
    def reset(self):
        self.pr.step()
        self.pr.stop()
        
    def setup(self,actions):

        self.pr.start()

        # Make cube for 
        self.obj_list = []
        self.block_order = []
        for i,a in enumerate(actions):
            col_idx = self.blocks.index(a[-1])
            self.block_order.append(a[-1])
            color = (cols.colors[col_idx][0:3]).tolist()

            pose = np.array(' '.join(a[0:-1].split(',')).split()).reshape(-1,3).astype(int)*0.05 + 0.05
            block_list = []
            for p in pose:
                obj = Shape.create(type=PrimitiveShape.CUBOID, 
                              color=color, size=[0.05, 0.05, 0.05],
                              position=p.tolist())
                obj.set_color(color)
                block_list.append(obj)
            self.obj_list.append(block_list)

        for j in range(20):
            self.pr.step()
        
        return

    def remove(self,piece):
        try:
            idx = self.block_order.index(piece)
            parts = self.obj_list[idx]
            for p in parts:
                p.remove()
        except:
            print('No such piece. Not Removed.')
        for j in range(2):
            self.pr.step()
        
        return self.check_moving()
            
    def check_moving(self):
        vels = []
        for p in self.obj_list:
            for block in p:
                try:
                    t,w = block.get_velocity()
                    vels.append(np.sum(t**2)+np.sum(w**2))
                            
                except:
                    vels.append(0)
        if np.sum(np.array(vels) > self.move_tolerance) > 1:
            self.collapsed = True
            return True 
        else:
            return False

def remove(sim,removal_list):
    
    collapsed = False
    count = 0
    for part in removal_list:
        collapsed = sim.remove(key[part])
        if collapsed:
            return collapsed, count
        else:
            count+=1
    return collapsed,count

def ga(sim,x,puzzle):
    
    best_count = 0
    iters = 0
    seq = x.copy()
    
    while iters < 30:
        sim.setup(puzzle)
        print('\rGa iter%d'%iters,end='')
        collapsed,count = remove(sim,seq)

        
        if count >= best_count:
            x = seq.copy()
            best_count = count
        
        if not collapsed:
            sim.reset()
            return x,iters 
        
        seq = x.copy()
        # randomly swap sequence
        bin_1 = best_count
        bin_2 = np.random.randint(count,7)

        temp = seq[bin_1]

        seq[bin_1] = seq[bin_2]
        seq[bin_2] = temp

        iters = iters+1
        sim.reset()
    return seq, iters

# Load puzzles
f = open("../soma_cube.txt", "r")
text = f.read()

split_sols = text.split('solution')
solutions = [split_sols[j] for j in range(1,241)]

solution_list = []
pic_list = []
for s in solutions:

    parts = s.split('\n')[1:8]
    solution_list.append(parts)

key = {'Orange':'V','Green':'L','Black':'b','Purple':'Z','White':'a','LBlue':'p','Blue':'T'}

iter_counts_seeds_tcn = []
iter_counts_seeds_sink = []
iter_counts_seeds_random = []
# Check collapses
for rep in range(1,100):

    extraction_orders_sink = np.load('../data/pred_order_%02d.npy'%rep)
    extraction_orders_tcn = np.load('../data/pred_order_tcn_%02d.npy'%rep)
    test_indices = np.load('../data/test_indices_%02d.npy'%rep).astype(int)
    test_indices_tcn = np.load('../data/test_indices_tcn_%02d.npy'%rep).astype(int)
    
    sim = BlockSim()

    print ('-----------Testing random init---------------')
    #print(test_indices==test_indices_tcn)

    # Genetic algorithm solution - random initialisation
    iter_counts = []
    for idx in test_indices.ravel():
        puzzle = solution_list[idx]
        init = np.array([k for k in key.keys()])
        np.random.shuffle(init)
        seq,iters = ga(sim,init,puzzle)
        iter_counts.append(iters)
        print('\r',len(iter_counts),end='')

    iter_counts_seeds_random.append(np.copy(iter_counts))

    print ('-----------Testing sink init---------------')
    # Genetic algorithm solution - sink initialisation
    iter_counts_sink = []
    for i,idx in enumerate(test_indices.ravel()):
        puzzle = solution_list[idx]
        init = extraction_orders_sink[i]
        seq,iters = ga(sim,init,puzzle)
        iter_counts_sink.append(iters)
        print('\r',len(iter_counts_sink),end='')

    iter_counts_seeds_sink.append(np.copy(iter_counts_sink))

    print ('-----------Testing tcn init---------------')
    # Genetic algorithm solution - tcn initialisation
    iter_counts_tcn = []
    for i,idx in enumerate(test_indices.ravel()):
        puzzle = solution_list[idx]
        init = extraction_orders_tcn[i]
        seq,iters = ga(sim,init,puzzle)
        iter_counts_tcn.append(iters)
        print('\r',len(iter_counts_tcn),end='')

    iter_counts_seeds_tcn.append(np.copy(iter_counts_tcn))

    print('Average iterations, rand: %2.2f +- %2.2f'%(np.mean(iter_counts),np.std(iter_counts)))
    print('Average iterations, tcn : %2.2f +- %2.2f'%(np.mean(iter_counts_tcn),np.std(iter_counts_tcn)))
    print('Average iterations, sink: %2.2f +- %2.2f'%(np.mean(iter_counts_sink),np.std(iter_counts_sink)))


    np.save('iters_tcn.npy',iter_counts_seeds_tcn)
    np.save('iters_sink.npy',iter_counts_seeds_sink)
    np.save('iters_random.npy',iter_counts_seeds_random)

    sim.pr.shutdown()

