import torch
import torchvision
from torchvision.transforms.functional import _get_inverse_affine_matrix, get_dimensions, InterpolationMode
from torchvision.transforms.functional_pil import affine as _affine

from typing import Optional, List

def affine(
    img: torch.Tensor,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    fill: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
) -> torch.Tensor:


    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    _, height, width =  get_dimensions(img)

    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    return _affine(img, matrix=matrix, interpolation=0, fill=fill)

def pre_image (matrix: torch.tensor, repeats: int, pad:int):

    """
    Enlarge the matrix image by a factor of repeats. Then embed in larger image
    """

    matrix = torch.repeat_interleave(matrix, repeats=5, dim=-1)
    matrix = torch.repeat_interleave(matrix, repeats=5, dim=-2)
    matrix = torch.nn.functional.pad(matrix, pad=(pad,)*4)
    return matrix


affine_ = torchvision.transforms.RandomAffine(
    degrees=(-30, 30),
    translate=None,
    scale=(.9, 1.1)
)

image = torch.rand((2**10, 1, 3, 3))
image = pre_image(image, 10, 10)
image = affine_(image)
image = torchvision.transforms.ToPILImage()(image[0])
# image.save('bye.png')
# image = affine(
#     img=image, 
#     angle=45,
#     translate=(0, 0),
#     scale=1,
#     shear=(0, 0),
#     center=(2*30 + 5*3,)*2,
#     )

image.save('hi.png')