# IA2C
ac_nets.py defines simple actor and critic for CNN networks, and associated functionalities. It requires make_pdtype from OpenAi's  baselines.common.distributions.

ac_lstm_nets.py defines actor and critic for LSTM networks, and associated functionalities. It requires make_pdtype from OpenAi's  baselines.common.distributions.

a2c_dec and a2c_lstm_dec are simple implementations of advantage actor critic learning for OpenAI gym environments with discrete actions and fixed episode lengths.