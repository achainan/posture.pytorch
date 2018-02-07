import argparse

parser = argparse.ArgumentParser(description='Train the pose estimation model.')
parser.add_argument('--num_epochs', type=int, default=5000)
parser.add_argument('--csv_dir', type=str, default='B/')
parser.add_argument('--root_dir', type=str, default='B/')
parser.add_argument('--input_height', type=int, default=128)
parser.add_argument('--model_name', type=str, default='posture')
parser.add_argument('--display_freq', type=int, default=20, help="Save preview to tensorboard ever X epochs")
parser.add_argument('--train_batch_size', type=int, default=120)
parser.add_argument('--val_batch_size', type=int, default=12)

display_parser = parser.add_mutually_exclusive_group(required=False)
display_parser.add_argument('--display', dest='display', action='store_true')
display_parser.add_argument('--no-display', dest='display', action='store_false')
parser.set_defaults(display=True)

normalize_parser = parser.add_mutually_exclusive_group(required=False)
normalize_parser.add_argument('--normalize', dest='normalize', action='store_true')
normalize_parser.add_argument('--no-normalization', dest='normalize', action='store_false')
parser.set_defaults(normalize=True)

cache_parser = parser.add_mutually_exclusive_group(required=False)
cache_parser.add_argument('--cached', dest='cached', action='store_true')
cache_parser.add_argument('--no-cached', dest='cached', action='store_false')
parser.set_defaults(cached=False)

args = parser.parse_args()