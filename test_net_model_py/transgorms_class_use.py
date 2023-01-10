from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "../virtual_env/PoinTr/lib/python3.8/site-packages/sklearn/datasets/images/china.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("tensor_img", tensor_img)

writer.close()