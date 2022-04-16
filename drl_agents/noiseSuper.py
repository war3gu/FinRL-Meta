
import numpy as np
import math
from typing import Iterable, List, Optional
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

class OrnsteinUhlenbeckActionNoiseSuper(OrnsteinUhlenbeckActionNoise):

    def __init__(
            self,
            mean: np.ndarray,
            sigma: np.ndarray,
            theta: float = 0.15,
            dt: float = 1e-2,
            initial_noise: Optional[np.ndarray] = None,
    ):
        self._sigma_base = sigma.copy()
        self.left = np.array([-1, -1])
        self.right = np.array([1, 1])
        super(OrnsteinUhlenbeckActionNoiseSuper, self).__init__(mean, sigma, theta, dt, initial_noise)

    def reset(self) -> None:
        super(OrnsteinUhlenbeckActionNoise, self).reset()

        self._sigma = self._sigma_base
        self.noise_prev = 2 * np.random.normal(size=self._mu.shape)
        #self._sigma *= 0.99   #gyk
        #print('_sigma ={0}'.format(self._sigma))

    def sigmaBaseMultiply(self, ratio) -> None:
        self._sigma_base *= ratio
        self._sigma = self._sigma_base.copy()
        print('_sigma_base ={0}'.format(self._sigma_base))

    def sigmaMultiply(self, ratio) -> None:
        self._sigma *= ratio
        #self._sigma = 0.1 * np.ones(2)
        #print('sigmaMultiply _sigma ={0}'.format(self._sigma))

    def setLeftRight(self, left, right) -> None:
        self.left = left
        self.right = right

    def __call__(self) -> np.ndarray:
        noise = (
                self.noise_prev * 0.7
                #+ self._theta * (-self.noise_prev) * self._dt
                + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )

        self._sigma *= 0.99
        self.noise_prev = noise
        return noise

    #暂时实现个均匀分布吧，噪音不能是无偏的，否则可能有大量的无效操作


class NormalActionNoiseSuper(NormalActionNoise):

    def reset(self) -> None:
        super(NormalActionNoise, self).reset()
        self._sigma *= 0.99   #gyk
        #print('_sigma ={0}'.format(self._sigma))

    def sigmaMultiply(self, ratio) -> None:
        self._sigma *= ratio
        #if self._sigma[0] < 0.01:
            #self._sigma = 0.00001 * np.ones(2)
        print('sigmaMultiply _sigma ={0}'.format(self._sigma))

    def __call__(self) -> np.ndarray:
        self._sigma *= 0.99
        return np.random.normal(self._mu, self._sigma)