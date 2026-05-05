import torch
import torch.nn as nn
import os

class MODEL(nn.Module):
    def __init__(self, generator):
        super(MODEL, self).__init__()
        self.generator = generator

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        output = self.generator(input)
        output = output.permute(0, 2, 3, 1)[0]
        return output

def convert_onnx(pt_model, fin_path, device, dummytens):
    model = MODEL(pt_model)
    model.to(device)
    model.eval()

    inputs = ['input']
    outputs = ['output']
    # dynamic_axes = {'input': {0: 'batches',
    #                           1: 'height',
    #                           2: 'width'},
    #                 'output': {0: 'batches',
    #                            1: 'height',
    #                            2: 'width'}}

    torch.onnx.export(model, dummytens, fin_path,
                      export_params=True, do_constant_folding=True,
                      # dynamic_axes=dynamic_axes,
                      input_names=inputs, output_names=outputs, opset_version=19,  # 14
                      verbose=False)

    print(f"✅ Сохранено. Размер: {os.path.getsize(fin_path) / 1024 ** 2:.1f} MB")

    model.train()