class AgentConfig(object):
  scale = 4000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 1
  #cnn_format = 'NCHW'
  cnn_format = 'NHWC'
  discount = 0.99
  target_q_update_step = 1 * scale

  learning_rate = 0.0005
  learning_rate_minimum = 0.0005
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = True
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

  max_ep_steps = 10000

  env_type = 'skip'
  action_repeat = 4

  img_scale = 255.
  clip = False

class EnvironmentConfig(object):
  env_name = 'MontezumaRevenge-v0'

  screen_width  = 84
  screen_height = 84


  max_reward = 1.
  min_reward = -1.

class PixelCNNConfig(AgentConfig, EnvironmentConfig):
  beta = 0.1
  psc_scale = 0.1

  psc_start = int(2.5 * AgentConfig.scale)

def get_config(args):
  config = PixelCNNConfig

  # if args.use_gpu:
  #   config.cnn_format = 'NHWC'
  # else:
  #   config.cnn_format = 'NCHW'

  return config
