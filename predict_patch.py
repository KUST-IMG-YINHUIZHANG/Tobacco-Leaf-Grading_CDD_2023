import os
import json
import xlrd
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms

from model import swin_tiny_patch4_window7_224 as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=8).to(device)
    # load model weights
    model_weight_path = "./weights/model.pth"

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    # pre-processing
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image and predict
    accu_all = 0
    sample_num = 0
    workBook = xlrd.open_workbook('./param_test_0705.xlsx')
    pic_path = "./tobacco_test_0705/"

    tobacco_class = [cla for cla in os.listdir(pic_path) if os.path.isdir(os.path.join(pic_path, cla))]
    for cla in tobacco_class:
        sheet = workBook.sheet_by_name(cla)
        cla_path = os.path.join(pic_path, cla)
        for parent, dirnames, filenames in os.walk(cla_path):
            accu_one = 0
            sample_num = len(filenames) + sample_num
            sample_num_one = len(filenames)

            for filename in tqdm(filenames):
                key = 0
                for name in sheet.col_values(0):
                    value = filename.split('.p')[0]
                    key += 1
                    if name == value:
                        break
                mainvein_thickness = []
                mainvein_thickness.append(sheet.cell_value(key-1, 1))
                mainvein_thickness = torch.tensor(mainvein_thickness)

                img_path = os.path.join(parent, filename)
                assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                img = Image.open(img_path)
                img = data_transform(img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)
                with torch.no_grad():
                    # predict
                    output = torch.squeeze(model(img.to(device), mainvein_thickness.to(device)[0])).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()
                predict_class = class_indict[str(predict_cla)]
                if predict_class == cla:
                    accu_one += 1
                    accu_all += 1
            print(cla, 'accu:', accu_one/sample_num_one)

    print('all accu:', accu_all/sample_num)

if __name__ == '__main__':
    main()
