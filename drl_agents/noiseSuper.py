
import numpy as np

from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

class OrnsteinUhlenbeckActionNoiseSuper(OrnsteinUhlenbeckActionNoise):

    def reset(self) -> None:
        super(OrnsteinUhlenbeckActionNoise, self).reset()
        self._sigma *= 0.99   #gyk
        print('_sigma ={0}'.format(self._sigma))

    def sigmaMultiply(self, ratio) -> None:
        self._sigma *= ratio
        print('sigmaMultiply _sigma ={0}'.format(self._sigma))


class NormalActionNoiseSuper(NormalActionNoise):

    def reset(self) -> None:
        super(NormalActionNoise, self).reset()
        self._sigma *= 0.99   #gyk
        print('_sigma ={0}'.format(self._sigma))

    def sigmaMultiply(self, ratio) -> None:
        self._sigma *= ratio
        #if self._sigma[0] < 0.01:
            #self._sigma = 0.00001 * np.ones(2)
        print('sigmaMultiply _sigma ={0}'.format(self._sigma))