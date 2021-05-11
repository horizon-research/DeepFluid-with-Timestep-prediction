import torch
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def create_ts_model():
    from models.TimeStepPreNetwork import TimestepPredictionNetwork
    """Returns an instance of the network for training and evaluation"""
    model = TimestepPredictionNetwork()
    return model


def read_ts_ckpt(path_to_pt, output):
    device = torch.device('cpu')
    model = create_ts_model()
    model.to(device)

    latest_checkpoint = torch.load(path_to_pt, map_location=torch.device('cpu'))
    model.load_state_dict(latest_checkpoint['model'])
    torch.save(model.state_dict(), output)


def create_pos_model():
    from models.PosCorrectionNetwork import PosCorrectionNetwork
    model = PosCorrectionNetwork()
    return model


def read_pos_ckpt(path_to_pt, output):
    device = torch.device('cpu')
    model = create_ts_model()
    model.to(device)

    latest_checkpoint = torch.load(path_to_pt, map_location=torch.device('cpu'))
    model.load_state_dict(latest_checkpoint['model'])
    torch.save(model.state_dict(), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate .pt for run_network.py from ckpt file')
    parser.add_argument('--ckpt', type=str, help='path to .pt ckpt file')
    parser.add_argument('--output', type=str, help='path to output .pt file')
    parser.add_argument('--modeltype', type=str, help='choose type of network')

    args = parser.parse_args()
    if args.modeltype == 'pos':
        read_pos_ckpt(args.ckpt, args.output)
    else:
        read_ts_ckpt(args.ckpt, args.output)