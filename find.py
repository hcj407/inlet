import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from compair import oblique_shock as obq
from compair import normal_shock as nos

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

gamma = 1.4
gm1 = gamma - 1 
gp1 = gamma + 1

def run(arg):
    '''
    Parameter
    ---------
    arg : namespace
    '''
    mode = arg.cmd


class Oswatitsch:
    def __init__(self, M0, N):
        self.M0 = M0
        self.N = N

    def oswatitsch(M0:float, t1:float, N:float):
        '''
        Parameter
        ---------
        M0 : 초기 마하수
        t1 : 쐐기형 램프의 첫번째 각도 [degree]
        N : 램프 단의 개수

        Return
        ------
        tpr : float
            총 전압력회복률
        thetas : array
            등강도 충격파를 갖는 램프의 각도
        stage : array

        '''
        M2, _, _, p0ratio, beta = obq.solve(M0, t1)

        thetas = [t1]
        stage = [{'stage':1, 'M1':M0, 'beta':beta, 'theta':t1, 'M2':M2,
                  'TPR':p0ratio}]
        
        M1, tpr = M2, p0ratio

        for i in range(1, N):
            # 등강도 조건: beta_{i+1} = arcsin(M0*sin(beta1) / M_{i+1})
            Mn = M0*np.sin(np.deg2rad(beta))
            if Mn <= 1.0: return None
            beta = np.rad2deg(np.arcsin(Mn/M1))  #deg
            if M1 <= 1.0: return None
            t = obq.theta_beta(beta, M1)  #deg
            M2, _, _, p0ratio, beta = obq.solve(M1, t)

            thetas.append(t)
            stage.append({'stage':i+1, 'M1':M1, 'beta':beta, 'theta':t, 'M2':M2,
                    'TPR':p0ratio})
            
            tpr *= p0ratio
            M1 = M2
        
        # 수직 종말 충격파
        if M2 <= 1.0: return None
        _, _, _, tpr_nos = nos.solve(M2)
        tpr *= tpr_nos


        return tpr, thetas, stage, M2, tpr_nos

    def find_max_pr(self, M0:float, N:int):
        def neg_tpr(t1):
            output = self.oswatitsch(M0, t1, N)
            if output is None:
                return np.inf
            return -output[0]

        bounds = (0.1, 89.9)
        grid = np.linspace(bounds[0], bounds[1], 500)
        vals = np.array([neg_tpr(t1) for t1 in grid])

        if not np.any(np.isfinite(vals)):
            raise ValueError("No feasible ramp angle found.")

        candidates = {int(np.nanargmin(vals))}
        for i in range(1, len(grid)-1):
            if vals[i] <= vals[i-1] and vals[i] <= vals[i+1]:
                candidates.add(i)

        best = None
        for i in candidates:
            left = grid[max(i-1, 0)]
            right = grid[min(i+1, len(grid)-1)]
            res = minimize_scalar(neg_tpr, bounds=(left, right), method='bounded')

            if not res.success or not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res

        if best is None:
            i = int(np.nanargmin(vals))
            t1 = grid[i]
        else:
            t1 = best.x

        return self.oswatitsch(M0, t1, N)