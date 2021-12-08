from dynaconf import Dynaconf
import configargparse
import toml

# Options for configuration
settings = configargparse.ArgParser(config_file_parser_class=configargparse.ConfigparserConfigFileParser,
                                    default_config_files=['identification_config.toml', 'file_path.toml', 'research_config.toml', 'nn_config.toml'])
settings.add_argument('--nn_config', type=toml.load, default='config/nn_config.toml')
settings.add_argument('--id', type=toml.load, default='config/identification_config.toml')
settings.add_argument('--research_config', type=toml.load, default='config/research_config.toml')
settings.add_argument('--file_config', type=toml.load, default='config/file_path.toml')

# Options for features
settings.add_argument('-t', '--temporal_feature', type=bool, default=False)
settings.add_argument('-l', '--link_feature', type=bool, default=False)
settings.add_argument('-n', '--neighbours_feature', type=bool, default=False)

# Options for models
settings.add_argument('--lstm', type=bool, default=False)
settings.add_argument('--rnn', type=bool, default=False)
settings.add_argument('--seq2seq', type=bool, default=False)
settings.add_argument('--lr', type=bool, default=False)
settings.add_argument('-s', '--separate', type=bool, default=False)

settings.add_argument('--save', type=bool, default=False)
args = settings.parse_args()
