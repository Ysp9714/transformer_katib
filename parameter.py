import argparse
from pathlib import Path
from models import HyperParameter


def get_arg():
    parser = argparse.ArgumentParser(description="Training Parameter")
    parser.add_argument(
        "-tfn",
        "--total_frame_num",
        type=int,
        default=600,
        help="total number of frames",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"{Path(__file__).parent}/datas",
        help="datas dir",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.15,
        help="label smoothing coefficient",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="change_data_shape,select_serial_random_frame",
    )
    parser.add_argument(
        "--frame_num", type=int, default=450, help="total number of frames"
    )
    parser.add_argument(
        "--label_num", type=int, default=2, help="total number of labels"
    )
    parser.add_argument("--feature_num", type=int, default=18, help="number of feature")
    parser.add_argument("--frame_filter_size", type=int, default=15)
    parser.add_argument("--frame_stride_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lstm_epochs", type=int, default=20)
    parser.add_argument("--second_epochs", type=int, default=10)
    parser.add_argument("--second_batch_size", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.8, help="validation ratio")
    parser.add_argument("--cut_frame", type=int, default=150, help="cutting frames")
    parser.add_argument("--training_step_num", type=int, default=2)
    param = HyperParameter(**vars(parser.parse_args()))
    return param