import torch
from torch.optim.optimizer import Optimizer
import copy

class AdaptivePSO(Optimizer):
    def __init__(self, params, swarm_size=20, inertia_weight=0.729, inertia_weight_strategy='constant',
                 c1=1.49445, c1_min=1.0, c1_max=2.0, c2=1.49445, c2_min=1.0, c2_max=2.0,
                 max_iter=1000, weight_decay=0, velocity_clamp=None, position_clamp=None,
                 threshold_low=0.2, threshold_high=0.5, c_adjust_step=0.05):
        defaults = dict(
            swarm_size=swarm_size,
            inertia_weight=inertia_weight,
            inertia_weight_strategy=inertia_weight_strategy,
            c1=c1, c1_min=c1_min, c1_max=c1_max,
            c2=c2, c2_min=c2_min, c2_max=c2_max,
            max_iter=max_iter,
            weight_decay=weight_decay,
            velocity_clamp=velocity_clamp,
            position_clamp=position_clamp,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            c_adjust_step=c_adjust_step
        )
        super(AdaptivePSO, self).__init__(params, defaults)
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.inertia_weight_strategy = inertia_weight_strategy
        self.c1 = c1
        self.c1_min = c1_min
        self.c1_max = c1_max
        self.c2 = c2
        self.c2_min = c2_min
        self.c2_max = c2_max
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.velocity_clamp = velocity_clamp
        self.position_clamp = position_clamp
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.c_adjust_step = c_adjust_step
        
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        for group in self.param_groups:
            group['particles'] = []
            group['velocities'] = []
            group['personal_best_positions'] = []
            group['personal_best_scores'] = []
            group['global_best_position'] = None
            group['global_best_score'] = float('inf')
            group['iteration'] = 0
            
            for _ in range(self.swarm_size):
                particle = []
                velocity = []
                for p in group['params']:
                    position = p.data.clone() + torch.randn_like(p.data) * 0.05
                    particle.append(position)
                    vel = torch.zeros_like(p.data)
                    velocity.append(vel)
                group['particles'].append(particle)
                group['velocities'].append(velocity)
                group['personal_best_positions'].append(copy.deepcopy(particle))
                group['personal_best_scores'].append(float('inf'))
    
    def step(self, loss_values):
        for group, loss in zip(self.param_groups, loss_values):
            
            if group['inertia_weight_strategy'] == 'linear':
                w_max = 0.9
                w_min = 0.4
                t = group['iteration']
                T = self.max_iter
                inertia_weight = w_max - ((w_max - w_min) * t / T)
            else:
                inertia_weight = group['inertia_weight']
            
            w = inertia_weight
            c1 = group['c1']
            c2 = group['c2']
            
            num_improved_particles = 0
            for i in range(self.swarm_size):
                
                loss = loss_values[i]
                if loss < group['personal_best_scores'][i]:
                    group['personal_best_scores'][i] = loss
                    group['personal_best_positions'][i] = [p.clone() for p in group['particles'][i]]
                    num_improved_particles += 1
                
                if loss < group['global_best_score']:
                    group['global_best_score'] = loss
                    group['global_best_position'] = [p.clone() for p in group['particles'][i]]
                
                for idx, p in enumerate(group['params']):
                    r1 = torch.rand_like(p.data)
                    r2 = torch.rand_like(p.data)
                    
                    cognitive_component = c1 * r1 * (group['personal_best_positions'][i][idx] - p.data)
                    social_component = c2 * r2 * (group['global_best_position'][idx] - p.data)
                    
                    group['velocities'][i][idx] = w * group['velocities'][i][idx] + cognitive_component + social_component
                    if self.velocity_clamp is not None:
                        group['velocities'][i][idx] = torch.clamp(
                            group['velocities'][i][idx],
                            min=self.velocity_clamp[0],
                            max=self.velocity_clamp[1]
                        )
                    
                    group['particles'][i][idx] = group['particles'][i][idx] + group['velocities'][i][idx]
                    
                    if self.position_clamp is not None:
                        group['particles'][i][idx] = torch.clamp(
                            group['particles'][i][idx],
                            min=self.position_clamp[0],
                            max=self.position_clamp[1]
                        )
            
            improvement_ratio = num_improved_particles / self.swarm_size
            
            if improvement_ratio < group['threshold_low']:
                group['c1'] = min(group['c1'] + group['c_adjust_step'], group['c1_max'])
                group['c2'] = max(group['c2'] - group['c_adjust_step'], group['c2_min'])
                if i == self.swarm_size - 2 :
                    print(f"Iteration {group['iteration']}: Increasing c1 to {group['c1']:.4f}, Decreasing c2 to {group['c2']:.4f}")
            
            elif improvement_ratio > group['threshold_high']:
                group['c1'] = max(group['c1'] - group['c_adjust_step'], group['c1_min'])
                group['c2'] = min(group['c2'] + group['c_adjust_step'], group['c2_max'])
                if i == self.swarm_size - 2 :
                    print(f"PSO-Iteration {group['iteration']}: Decreasing c1 to {group['c1']:.4f}, Increasing c2 to {group['c2']:.4f}")
            
            group['iteration'] += 1
        
        for group in self.param_groups:
            if group['global_best_position'] is not None:
                with torch.no_grad():
                    for p, best_p in zip(group['params'], group['global_best_position']):
                        p.data.copy_(best_p)
        
        return min(group['global_best_score'] for group in self.param_groups)
