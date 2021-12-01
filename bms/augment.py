import albumentations as A
from albumentations.augmentations.geometric.functional import safe_rotate_enlarged_img_size, _maybe_process_in_chunks
import cv2
import numpy as np


def expand_safe_rotate(
    img: np.ndarray,
    angle: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    value: int = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
):

    old_rows, old_cols = img.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (old_cols / 2, old_rows / 2)

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    # Rotation Matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Shift the image to create padding
    rotation_mat[0, 2] += new_cols / 2 - image_center[0]
    rotation_mat[1, 2] += new_rows / 2 - image_center[1]

    # CV2 Transformation function
    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=rotation_mat,
        dsize=(new_cols, new_rows),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    # rotate image with the new bounds
    rotated_img = warp_affine_fn(img)

    return rotated_img


class ExpandSafeRotate(A.SafeRotate):

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(ExpandSafeRotate, self).__init__(
            limit=limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            p=p)

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return expand_safe_rotate(
            img=img, value=self.value, angle=angle, interpolation=interpolation, border_mode=self.border_mode
        )

    
class CropWhite(A.DualTransform):
    
    def __init__(self, value=(255, 255, 255), pad=0):
        super(CropWhite, self).__init__(always_apply=True)
        self.value = value
        self.pad = pad
        assert pad >= 0
        
    def apply(self, img, **params):
        height, width, _ = img.shape
        x = (img != self.value).sum(axis=2)
        if x.sum() == 0:
            return img
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top+1 < height:
            top += 1
        # top = max(0, top - self.pad)
        bottom = height
        while row_sum[bottom-1] == 0 and bottom-1 > top:
            bottom -= 1
        # bottom = min(height, bottom + self.pad)
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left+1 < width:
            left += 1
        # left = max(0, left - self.pad)
        right = width
        while col_sum[right-1] == 0 and right-1 > left:
            right -= 1
        # right = min(width, right + self.pad)
        img = img[top:bottom, left:right]
        if self.pad > 0:
            img = A.augmentations.pad_with_params(img, self.pad, self.pad, self.pad, self.pad,
                                                  border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def get_transform_init_args_names(self):
        return ('value', 'pad')

    
class ResizePad(A.DualTransform):

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, value=(255, 255, 255)):
        super(ResizePad, self).__init__(always_apply=True)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.value = value

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w, _ = img.shape
        img = A.augmentations.geometric.functional.resize(
            img, 
            height=min(h, self.height), 
            width=min(w, self.width), 
            interpolation=interpolation
        )
        h, w, _ = img.shape
        pad_top = (self.height - h) // 2
        pad_bottom = (self.height - h) - pad_top
        pad_left = (self.width - w) // 2
        pad_right = (self.width - w) - pad_left
        img = A.augmentations.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=cv2.BORDER_CONSTANT,
            value=self.value,
        )
        return img
