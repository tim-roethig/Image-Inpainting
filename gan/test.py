import matplotlib.pyplot as plt
import torch

from model import InpaintGenerator
from prep_data.prep_data import PrepData


if __name__ == '__main__':
    device = torch.device('cpu')

    model = InpaintGenerator([1, 2, 4, 8], 2).double()
    model.load_state_dict(torch.load('gan_generator'))
    model = model.to(device)
    model.eval()

    img, mask, gt_img = PrepData()[3]
    mask = mask[0, :, :]
    mask = mask[None, :, :]

    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    img.unsqueeze_(0)
    gt_img.unsqueeze_(0)
    mask.unsqueeze_(0)
    print(img)

    with torch.no_grad():
        output = model(img.to(device), mask.to(device))
    print(output)
    mask.to(device)
    img.to(device)
    output.to(device)
    output = (mask * img) + ((1 - mask) * output)
    print(output)

    plt.imshow(output[0].permute(1, 2, 0))
    plt.show()