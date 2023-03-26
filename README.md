## README

1. Download the dataset: https://pan.baidu.com/s/1Vr25cn2DTN7KcjwcMpVIWg    key:qzcv
2. Set `--data-path` in the `train.py` script to the absolute path of the unpacked data folder.
3. Set the `--weights` parameter in the `train.py` script to the path of the pre-trained weights
4. Import the same model in the `predict.py` script as in the training script, and set the `model_weight_path` to the path of the trained model weights (the default is saved in the weights folder)
5. Set `img_path` in the `predict.py` script to the absolute path of the image you want to predict
6. Set the weight path `workBook` and the 'pic_path' path to use the `predict_patch.py` script for the `predict_patch.py`
