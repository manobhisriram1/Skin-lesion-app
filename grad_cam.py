import torch
import cv2
import numpy as np

# -------------------------
# ✅ Grad-CAM
# -------------------------
def generate_gradcam(model, img_tensor, target_layer):
    model.zero_grad()
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    output[0, pred_class].backward()

    handle1.remove()
    handle2.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = features[0]

    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# -------------------------
# ✅ Grad-CAM++
# -------------------------
def generate_gradcam_plus_plus(model, img_tensor, target_layer):
    model.zero_grad()
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    output[0, pred_class].backward(retain_graph=True)

    handle1.remove()
    handle2.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = features[0]

    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# -------------------------
# ✅ Overlay Heatmap
# -------------------------
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay
