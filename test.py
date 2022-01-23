import matplotlib.pyplot as plt
import torch
import numpy as np
import time

from model import PartialConvNet
from prep_data_efficient_lines_testing import PrepData

start = time.time()

device = torch.device('cpu')

model = PartialConvNet().double()
model.load_state_dict(torch.load('model_epoch_13_rectangles.t7', map_location=device))
model = model.to(device)
model.eval()

#rand = np.random.randint(0, 159)
img, mask, gt_img = PrepData()[0]

#plt.imshow(img.permute(1, 2, 0))
# See ground truth images
#plt.imsave(f'test{rand}.png', gt_img.permute(1, 2, 0).numpy())

img.unsqueeze_(0)
gt_img.unsqueeze_(0)
mask.unsqueeze_(0)
#print(img)

with torch.no_grad():
    output = model(img.to(device), mask.to(device))
#print(output)
output = (mask * img) + ((1 - mask) * output)
#print(output)

output = output[0].permute(1, 2, 0).numpy()
# Make sure output is always in range 0..1
output = np.interp(output, (output.min(), output.max()), (0., 1.))
#print(output)

#plt.imshow(output[0].permute(1, 2, 0))
plt.imsave('test_efficiency_new.png', output)
#plt.show()
end = time.time()

print("The time of execution of the test is: ", end-start)