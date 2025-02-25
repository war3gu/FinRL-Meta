# common library
import random
import time

import numpy as np
import pandas as pd
from finrl.apps import config
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.finrl_meta.preprocessor.preprocessors import data_split
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from drl_agents.noiseSuper import *
from gym import spaces
# RL models from stable-baselines


MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "normal_super": NormalActionNoiseSuper,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
    "ornstein_uhlenbeck_super":OrnsteinUhlenbeckActionNoiseSuper,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

        self.earn_mean = -10
        self.earn_max = -10
        self.earn_min = -10
        self.earn_median = -10

        self.earn1_mean = -10
        self.sigma_base = 1
        self.static_count = 0
        self.big_sigma_count = 0

        self.earn_list = []

        self.need_reset = False

        self.gradient_steps_old = 100

        self.sigma_v = None

        #self.sigma_list_collect_scratch = [40, 30, 20, 10, 8, 6, 4, 2]   #scratch对应的动作完全随机
        #self.sigma_list_collect = [40, 30, 20, 10, 8, 6, 4, 2, 1, 0.5]

        #self.sigma_list_collect_scratch = [9, 3, 1, 0.8, 0.5, 0.3, 0.2, 0.1]      #可能这组参数只对8*8的网络有效
        #self.sigma_list_collect = [9, 3, 1, 0.8, 0.5, 0.3, 0.2, 0.1]                    #训练的动作是actor产生的，noise不能太离谱

        #sigma太多，似乎就拟合不了了

        #self.theta_list = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        e = 2.72

        self.sigma_list_collect_scratch = [e*3, e*2, e, 1/e, 0.1]
        self.sigma_list_collect = [e*3, e*2, e, 1/e, 0.1]
        #self.sigma_list_collect_scratch = [0.5]
        #self.sigma_list_collect = [0.5]
        self.theta_list_scratch = [0.0001]
        self.theta_list = [0.0001]

    def change_noise(self) -> None:
        #随机的设置一个sigma，theta
        #

        num_timesteps = self.locals['self'].num_timesteps
        learning_starts = self.locals['self'].learning_starts
        sigma = None
        theta = None
        noise_end = 15000000
        '''
        if num_timesteps <= learning_starts:
            sigma = random.sample(self.sigma_list_collect_scratch, 1)
            theta = random.sample(self.theta_list_scratch, 1)
            print('scratch')
        elif num_timesteps <= learning_starts + noise_end:
            sigma = random.sample(self.sigma_list_collect, 1)
            theta = random.sample(self.theta_list, 1)
            print('before noise_end')
        else:
            sigma = [0.1]
            theta = [0.0001]
            print('after noise_end')
        '''
        if num_timesteps <= learning_starts:
            sigma = [1]
        else:
            sigma = [1]

        '''
        elif num_timesteps <= learning_starts*2:
            sigma = [10]
        elif num_timesteps <= learning_starts*3:
            sigma = [5]
        elif num_timesteps <= learning_starts*4:
            sigma = [1]
        elif num_timesteps <= learning_starts*5:
            sigma = [0.6]
        elif num_timesteps <= learning_starts*6:
            sigma = [0.5]
        elif num_timesteps <= learning_starts*7:
            sigma = [0.4]
        elif num_timesteps <= learning_starts*8:
            sigma = [0.3]
        elif num_timesteps <= learning_starts*9:
            sigma = [0.2]
        '''

        theta = [100]


        print('sigma = {0}'.format(sigma[0]))
        print('theta = {0}'.format(theta[0]))
        self.model.action_noise.setSigma(sigma[0])
        self.model.action_noise.setTheta(theta[0])


        action_noise_list = [self.locals['action_noise']]#.noises
        for index, everyOne in enumerate(action_noise_list):
            action_noise = action_noise_list[index]
            action_noise.setSigma(sigma[0])
            action_noise.setTheta(theta[0])



    def _on_rollout_start(self) -> None:
        print('_on_rollout_start in TensorboardCallback')

    def _on_rollout_end(self) -> None:
        print('_on_rollout_end in TensorboardCallback')   #开始训练
        #通过统计self.earn_list里面正的个数， 设置gradient_steps来决定是否训练。mean越大越有训练价值



    def _on_training_start(self) -> None:
        self.logger.log("train/start")
        self.logger.dump()
        #print('_on_training_start in TensorboardCallback')
        #learn once，change noise once
        #self.model.action_noise.sigmaBaseMultiply(0.9)       #learn的时候修改base


    def _on_training_end(self) -> None:
        self.logger.log("train/end")
        self.logger.dump()

    def _on_step(self) -> bool:
        #try:

        '''

        num_timesteps = self.locals['self'].num_timesteps
        learning_starts = self.locals['learning_starts']

        episode_end = False

        earn1_list = []
        infos = self.locals['infos']
        for index, everyOne in enumerate(infos):
            earn1 = everyOne['earn1']
            if earn1:
                episode_end = True
                self.earn_list.append(earn1)
                earn1_list.append(earn1)
                print('earn1 = {0}'.format(earn1))

        if episode_end:
            self.change_noise()
        '''
            #为什么网络大了一点就训练不起来了？因为初始化的输出就变了，旧的sigma不再起作用

        #self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        #self.logger.dump()

        #except BaseException:
            #self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions)#, theta=100 #, dt=0.01    #theta越小干扰越大
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
            reset_num_timesteps=False,
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment):
        test_env, _ = environment.get_sb_env()
        """make a prediction"""
        account_memory = pd.DataFrame()
        actions_memory = pd.DataFrame()
        earn_memory    = []
        #test_env.reset()
        #通过env_method得到总长度，然后再循环
        test_obs = test_env.reset()
        sl = test_env.env_method(method_name="get_step_length")

        path_length = test_env.env_method(method_name="get_path_length")

        for k in range(path_length[0]):
            test_env.env_method(method_name="reset_memory")   #每个path执行一遍
            for i in range(sl[0]):
                action, _states = model.predict(test_obs)
                test_obs, rewards, dones, info = test_env.step(action)
                if dones[0]:
                    print("hit one path end!")
                    aaa = test_env.env_method(method_name="save_asset_memory")
                    bbb = test_env.env_method(method_name="save_action_memory")
                    earn = test_env.env_method(method_name="get_earn")
                    account_memory = account_memory.append(aaa[0])
                    actions_memory = actions_memory.append(bbb[0])
                    earn_memory.append(earn[0])
                    break
        return account_memory, actions_memory, earn_memory

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_total_asset)
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                environment.cash
                + (environment.price_array[environment.time] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets


class DRLEnsembleAgent:
    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        seed=None,
        verbose=1,
    ):

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(temp_model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )
        return model

    @staticmethod
    def train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps//1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(
            "results/account_value_validation_{}_{}.csv".format(model_name, iteration)
        )
        sharpe = (
            (4 ** 0.5)
            * df_total_value["daily_return"].mean()
            / df_total_value["daily_return"].std()
        )
        return sharpe

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
    ):

        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for i in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(
        self, model, name, last_state, iter_num, turbulence_threshold, initial
    ):
        """make a prediction based on trained model"""

        ## trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    trade_data,
                    self.stock_dim,
                    self.hmax,
                    self.initial_amount,
                    self.buy_cost_pct,
                    self.sell_cost_pct,
                    self.reward_scaling,
                    self.state_space,
                    self.action_space,
                    self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(
            "results/last_state_{}_{}.csv".format(name, i), index=False
        )
        return last_state

    def run_ensemble_strategy(
        self, A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
    ):
        """Ensemble Strategy that combines PPO, A2C and DDPG"""
        print("============Start Ensemble Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        ppo_sharpe_list = []
        ddpg_sharpe_list = []
        a2c_sharpe_list = []

        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            ## initial state is empty
            if i - self.rebalance_window - self.validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter (63 days)
            end_date_index = self.df.index[
                self.df["date"]
                == self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ]
            ].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            historical_turbulence = self.df.iloc[
                start_date_index : (end_date_index + 1), :
            ]

            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            # print(historical_turbulence_mean)

            if historical_turbulence_mean > insample_turbulence_threshold:
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            ############## Environment Setup starts ##############
            ## training env
            train = data_split(
                self.df,
                start=self.train_period[0],
                end=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            self.train_env = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        train,
                        self.stock_dim,
                        self.hmax,
                        self.initial_amount,
                        self.buy_cost_pct,
                        self.sell_cost_pct,
                        self.reward_scaling,
                        self.state_space,
                        self.action_space,
                        self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )

            validation = data_split(
                self.df,
                start=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
                end=self.unique_trade_date[i - self.rebalance_window],
            )
            ############## Environment Setup ends ##############

            ############## Training and Validation starts ##############
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
            # print("==============Model Training===========")
            print("======A2C Training========")
            model_a2c = self.get_model(
                "a2c", self.train_env, policy="MlpPolicy", model_kwargs=A2C_model_kwargs
            )
            model_a2c = self.train_model(
                model_a2c,
                "a2c",
                tb_log_name="a2c_{}".format(i),
                iter_num=i,
                total_timesteps=timesteps_dict["a2c"],
            )  # 100_000

            print(
                "======A2C Validation from: ",
                validation_start_date,
                "to ",
                validation_end_date,
            )
            val_env_a2c = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        validation,
                        self.stock_dim,
                        self.hmax,
                        self.initial_amount,
                        self.buy_cost_pct,
                        self.sell_cost_pct,
                        self.reward_scaling,
                        self.state_space,
                        self.action_space,
                        self.tech_indicator_list,
                        turbulence_threshold=turbulence_threshold,
                        iteration=i,
                        model_name="A2C",
                        mode="validation",
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )
            val_obs_a2c = val_env_a2c.reset()
            self.DRL_validation(
                model=model_a2c,
                test_data=validation,
                test_env=val_env_a2c,
                test_obs=val_obs_a2c,
            )
            sharpe_a2c = self.get_validation_sharpe(i, model_name="A2C")
            print("A2C Sharpe Ratio: ", sharpe_a2c)

            print("======PPO Training========")
            model_ppo = self.get_model(
                "ppo", self.train_env, policy="MlpPolicy", model_kwargs=PPO_model_kwargs
            )
            model_ppo = self.train_model(
                model_ppo,
                "ppo",
                tb_log_name="ppo_{}".format(i),
                iter_num=i,
                total_timesteps=timesteps_dict["ppo"],
            )  # 100_000
            print(
                "======PPO Validation from: ",
                validation_start_date,
                "to ",
                validation_end_date,
            )
            val_env_ppo = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        validation,
                        self.stock_dim,
                        self.hmax,
                        self.initial_amount,
                        self.buy_cost_pct,
                        self.sell_cost_pct,
                        self.reward_scaling,
                        self.state_space,
                        self.action_space,
                        self.tech_indicator_list,
                        turbulence_threshold=turbulence_threshold,
                        iteration=i,
                        model_name="PPO",
                        mode="validation",
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )
            val_obs_ppo = val_env_ppo.reset()
            self.DRL_validation(
                model=model_ppo,
                test_data=validation,
                test_env=val_env_ppo,
                test_obs=val_obs_ppo,
            )
            sharpe_ppo = self.get_validation_sharpe(i, model_name="PPO")
            print("PPO Sharpe Ratio: ", sharpe_ppo)

            print("======DDPG Training========")
            model_ddpg = self.get_model(
                "ddpg",
                self.train_env,
                policy="MlpPolicy",
                model_kwargs=DDPG_model_kwargs,
            )
            model_ddpg = self.train_model(
                model_ddpg,
                "ddpg",
                tb_log_name="ddpg_{}".format(i),
                iter_num=i,
                total_timesteps=timesteps_dict["ddpg"],
            )  # 50_000
            print(
                "======DDPG Validation from: ",
                validation_start_date,
                "to ",
                validation_end_date,
            )
            val_env_ddpg = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        validation,
                        self.stock_dim,
                        self.hmax,
                        self.initial_amount,
                        self.buy_cost_pct,
                        self.sell_cost_pct,
                        self.reward_scaling,
                        self.state_space,
                        self.action_space,
                        self.tech_indicator_list,
                        turbulence_threshold=turbulence_threshold,
                        iteration=i,
                        model_name="DDPG",
                        mode="validation",
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )
            val_obs_ddpg = val_env_ddpg.reset()
            self.DRL_validation(
                model=model_ddpg,
                test_data=validation,
                test_env=val_env_ddpg,
                test_obs=val_obs_ddpg,
            )
            sharpe_ddpg = self.get_validation_sharpe(i, model_name="DDPG")

            ppo_sharpe_list.append(sharpe_ppo)
            a2c_sharpe_list.append(sharpe_a2c)
            ddpg_sharpe_list.append(sharpe_ddpg)

            print(
                "======Best Model Retraining from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window],
            )
            # Environment setup for model retraining up to first trade date
            # train_full = data_split(self.df, start=self.train_period[0], end=self.unique_trade_date[i - self.rebalance_window])
            # self.train_full_env = DummyVecEnv([lambda: StockTradingEnv(train_full,
            #                                                    self.stock_dim,
            #                                                    self.hmax,
            #                                                    self.initial_amount,
            #                                                    self.buy_cost_pct,
            #                                                    self.sell_cost_pct,
            #                                                    self.reward_scaling,
            #                                                    self.state_space,
            #                                                    self.action_space,
            #                                                    self.tech_indicator_list,
            #                                                    print_verbosity=self.print_verbosity)])
            # Model Selection based on sharpe ratio
            if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
                model_use.append("PPO")
                model_ensemble = model_ppo

                # model_ensemble = self.get_model("ppo",self.train_full_env,policy="MlpPolicy",model_kwargs=PPO_model_kwargs)
                # model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['ppo']) #100_000
            elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
                model_use.append("A2C")
                model_ensemble = model_a2c

                # model_ensemble = self.get_model("a2c",self.train_full_env,policy="MlpPolicy",model_kwargs=A2C_model_kwargs)
                # model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['a2c']) #100_000
            else:
                model_use.append("DDPG")
                model_ensemble = model_ddpg

                # model_ensemble = self.get_model("ddpg",self.train_full_env,policy="MlpPolicy",model_kwargs=DDPG_model_kwargs)
                # model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['ddpg']) #50_000

            ############## Training and Validation ends ##############

            ############## Trading starts ##############
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            # print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction(
                model=model_ensemble,
                name="ensemble",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )
            ############## Trading ends ##############

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                model_use,
                a2c_sharpe_list,
                ppo_sharpe_list,
                ddpg_sharpe_list,
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            "Model Used",
            "A2C Sharpe",
            "PPO Sharpe",
            "DDPG Sharpe",
        ]

        return df_summary
