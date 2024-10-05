import numpy as np


class SimpleShow:
    
    def __init__(self, losses: str, G = 'G-Loss', D = 'D-Loss' ):
        
        losses = losses.split('\n')
        losses.pop(0)
        losses.pop()
        self.losses = losses
        self.G = G
        self.D = D
        
        self.__get_abs_loss__()
    
    
    def __calc_loss__(self):
        
        all_ = dict()
        all_[self.G] = list()
        all_[self.D] = list()
        for any_loss in self.losses:
            any_loss = any_loss.split(',')
            for loss in any_loss:
                loss = loss.split(' ')
                if 'G' in loss:
                    all_[self.G].append(float(loss[loss.index('G')+2]))
                elif 'D' in loss:
                    all_[self.D].append(float(loss[loss.index('D')+2]))
        
        return all_
    
    
    def __get_abs_loss__(self):
        
        loss = self.__calc_loss__()
        g_losses = []
        d_losses = []
        for key , val in loss.items():
            for any_val in val:
                if key == self.G:
                    g_losses.append(any_val)
                elif key == self.D:
                    d_losses.append(any_val)
        
        self.G = g_losses
        self.D = d_losses
    
    
    def get_loss(self):
        
        g_mae = np.mean(np.abs(self.G))
        g_mse = np.mean(np.square(self.G))
        d_mae = np.mean(np.abs(self.D))
        d_mse = np.mean(np.square(self.D))
        
        return g_mae, g_mse, d_mae, d_mse
    
    
    def show(self):
        
        g_mae, g_mse, d_mae, d_mse = self.get_loss()
        print("The Mean Abselute Error of Generator is: ", g_mae)
        print("The Mean Squared Error of Generator is: ", g_mse)
        print("The Mean Absolute Error of Discriminator is: ", d_mae)
        print("The Mean Squared Error of Discriminator is: ", d_mse)
        
        return
    
#cloner174