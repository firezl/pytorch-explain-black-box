import torch
from torchvision import models
import cv2
import sys
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# 在新版PyTorch中，直接使用dtype而不是*Tensor
Tensor = torch.FloatTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img).to(device)
    preprocessed_img_tensor.unsqueeze_(0)
    return preprocessed_img_tensor.requires_grad_(False)


def save_image(filename, img_array):
    """安全地保存图像文件"""
    try:
        # 确保图像是无符号8位整数类型
        if img_array.dtype != np.uint8:
            img_array = (255 * img_array).clip(0, 255).astype(np.uint8)

        # 保存图像
        success = cv2.imwrite(filename, img_array)
        if not success:
            print(f"保存{filename}失败")
    except Exception as e:
        print(f"保存{filename}时出错: {e}")


def save(mask, img, blurred):
    # 提取掩码数据并进行归一化处理
    mask = mask.cpu().detach().numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask

    # 处理单通道掩码
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    # 创建热力图可视化
    mask_uint8 = (255 * mask).astype(np.uint8)
    try:
        # 尝试应用颜色映射
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    except Exception as e:
        print(f"应用颜色映射时出错: {e}")
        # 创建一个备用热力图
        heatmap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # 简单颜色映射：红色表示高值，蓝色表示低值
        heatmap[:, :, 0] = (255 * (1 - mask)).astype(np.uint8)  # 蓝色通道
        heatmap[:, :, 2] = (255 * mask).astype(np.uint8)  # 红色通道

    # 计算CAM（Class Activation Map）
    heatmap = np.float32(heatmap) / 255
    img_float = np.float32(img) / 255
    cam = 1.0 * heatmap + img_float
    cam = cam / np.max(cam)

    # 计算扰动图像
    if len(mask.shape) == 2:
        # 扩展单通道掩码以匹配RGB图像
        mask_expanded = np.expand_dims(mask, axis=2)
        mask_expanded = np.repeat(mask_expanded, 3, axis=2)
        perturbated = np.multiply(1 - mask_expanded, img_float) + np.multiply(
            mask_expanded, blurred
        )
    else:
        perturbated = np.multiply(1 - mask, img_float) + np.multiply(mask, blurred)

    # 保存所有结果图像
    perturbated_save = (255 * perturbated).astype(np.uint8)
    heatmap_save = (255 * heatmap).astype(np.uint8)
    cam_save = (255 * cam).astype(np.uint8)

    if len(mask.shape) == 2:
        # 对于2D掩码，将其转换为3通道以保存
        mask_save = np.stack([mask, mask, mask], axis=2)
        mask_save = (255 * mask_save).astype(np.uint8)
    else:
        mask_save = (255 * mask).astype(np.uint8)

    # 使用安全的保存函数
    save_image("perturbated.png", perturbated_save)
    save_image("heatmap.png", heatmap_save)
    save_image("mask.png", mask_save)
    save_image("cam.png", cam_save)


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        # 创建一个新的数组
        output = np.expand_dims(np.float32(img), axis=0)
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    output = output.to(device)
    output.unsqueeze_(0)
    output.requires_grad_(requires_grad)
    return output


def load_model():
    try:
        # 尝试使用新版API
        from torchvision.models import VGG19_Weights

        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    except ImportError:
        # 兼容旧版API
        model = models.vgg19(pretrained=True)

    model.eval()
    model = model.to(device)

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return model


if __name__ == "__main__":
    # Hyper parameters.
    # TBD: Use argparse
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2

    model = load_model()
    original_img = cv2.imread(sys.argv[1], 1)
    if original_img is None:
        print(f"无法读取图像文件: {sys.argv[1]}")
        sys.exit(1)
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img) / 255
    # 处理模糊效果
    blurred_img1 = cv2.GaussianBlur(original_img.astype(np.float32) / 255, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype=np.float32)

    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_torch(mask_init)

    # 使用更现代的上采样方法
    upsample = torch.nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
    upsample = upsample.to(device)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    target = torch.nn.Softmax(dim=1)(model(img))
    category = np.argmax(target.cpu().detach().numpy())
    print("Category with highest probability", category)
    print("Optimizing.. ")

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = upsampled_mask.expand(
            1, 3, upsampled_mask.size(2), upsampled_mask.size(3)
        )

        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + blurred_img.mul(
            1 - upsampled_mask
        )

        # 使用NumPy生成随机噪声而不是cv2.randn
        noise = np.random.normal(0, 0.2, (224, 224, 3)).astype(np.float32)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise

        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))
        loss = (
            l1_coeff * torch.mean(torch.abs(1 - mask))
            + tv_coeff * tv_norm(mask, tv_beta)
            + outputs[0, category]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    upsampled_mask = upsample(mask)
    save(upsampled_mask, original_img, blurred_img_numpy)
