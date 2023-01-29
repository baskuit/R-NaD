import torch
import torchvision
import PIL
from functorch import vmap
from random import random


def raw_obs(batch_size, h, w):
    return torch.rand(size=(batch_size, 1, h, w)).to(torch.float)


def scale(x: torch.Tensor, h=1, w=None):
    if w is None:
        w = h
    return x.repeat_interleave(h, dim=-2).repeat_interleave(w, dim=-1)


def random_noise(x, std=0.1):
    return x + std * torch.randn_like(x)


def obs(
    batch_size,
    h,
    w,
    h_scale=5,
    w_scale=None,
):
    x = raw_obs(batch_size, h, w)
    x = scale(x, h_scale, w_scale)
    x = torch.nn.functional.pad(x, (10,) * 4)
    x = random_noise(x, std=0.1)
    return x


random_affine = torchvision.transforms.RandomAffine(
    degrees=(-30, 30),
    shear=(0.8, 1.2),
)

image = torchvision.transforms.ToPILImage()

x = raw_obs(2, 3, 3)
x = scale(x, 5)
x = torch.nn.functional.pad(x, (10, 10, 10, 10))


def affine(
    img,
    angle,
):
    return torchvision.transforms.functional.affine(
        img=img,
        angle=angle,
        translate=(0, 0),
        scale=1,
        shear=1,
    )


batch_affine = vmap(affine)
batch_size = 10
x = obs(batch_size, 3, 3)
y = batch_affine(
    x,
    torch.rand((batch_size,), dtype=torch.float),
)

for _ in range(batch_size):
    z = image(y[_, 0])
    z.save(f"{_}.png")

exit()
x = random_affine(x)
x = random_noise(x)
x = torch.clamp(x, 0, 1)


y = image(x[0, 0])
y.show()
y.save("test.png")

z = image(x[1, 0])
z.save("test2.png")
