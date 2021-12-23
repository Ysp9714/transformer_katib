import math
from pathlib import Path
from typing import Any, Dict, Tuple, List
import tensorflow as tf
from pydantic import BaseModel

__slot__ =["HyperParameter", "TrainingDataset"]

class TrainingDataset(BaseModel):
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset

    class Config:
        arbitrary_types_allowed = True


class Singleton(type):
    _instance: Dict[str,object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super().__call__(*args, **kwargs)
        return cls._instance[cls]


class HyperParameter(metaclass=Singleton):
    def __init__(
        self,
        model_name="short_long_model/",
        obj_name="blister",
        saved_model_folder="./model",
        data_path="/home/ubuntu/object-detection-training/scripts/postprocessing/data/action_recognition_data/",
        epochs=150,
        lstm_epochs=30,
        label_smoothing=0.15,
        val_ratio=0.8,
        total_frame_num=600,
        frame_num=500,
        feature_num=18,
        frame_filter_size=15,
        frame_stride_size=5,
        batch_size=256,
        training_step_num=2,
        second_epochs=40,
        second_batch_size=64,
        label_num=3,
        augmentations="change_data_shape,select_serial_random_frame",
        cut_frame=150,
    ):
        # Settings
        self.model_name = model_name
        self.saved_model_folder:Path = Path(saved_model_folder)
        self.obj_name = obj_name
        self.data_path = Path(data_path)
        # Hyperparameter
        self._label_num = label_num
        self.lstm_epochs: int = lstm_epochs
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.val_ratio: float = val_ratio
        self.total_frame_num: int = total_frame_num
        self.training_step_num: int = training_step_num

        self.label_smoothing = label_smoothing
        self._frame_num: int = frame_num
        self.feature_num: int = feature_num

        self.frame_filter_size: int = frame_filter_size
        self.frame_stride_size: int = frame_stride_size

        self.second_epochs:int = second_epochs
        self.second_batch_size:int = second_batch_size

        self._argmentations = augmentations
        # Maked Feature
        self.label:list = []
        self.train_data_size: int = 1
        self.val_data_size: int = 1
        # For Test
        self.cut_frame = cut_frame

    @property
    def frame_num(self)->int:
        if self.training_step_num==1:
            return self.total_frame_num
        return self._frame_num
    @property
    def augmentations(self)->List[str]:
        str_augs = self._argmentations.replace(' ','')
        args = str_augs.split(',')
        if args[0]!='change_data_shape':
            args.insert(0,'change_data_shape')
        return args 

    @augmentations.setter
    def augmentations(self, str_augs)->None:
        self._argmentations = str_augs

    @property
    def label_num(self)->int:
        return self._label_num -1

    @label_num.setter
    def label_num(self, num):
        self._label_num = num

    @property
    def channel_num(self)->int:
        return math.ceil(
            (self.total_frame_num - self.frame_filter_size) / self.frame_stride_size + 1
        )

    @property
    def long_input_shape(self)->Tuple:
        return (self.frame_num, self.feature_num, 1)

    @property
    def short_input_shape(self)->Tuple:
        return (self.feature_num, 1)

    @property
    def model_save_folder_path(self)->Path:
        return self.saved_model_folder/self.obj_name

    @property
    def model_path(self)->Path:
        return (
            self.model_save_folder_path/"weights.{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}.hdf5"
        )
    @property
    def output_shape(self):
        return len(self.label)