import torch

def predict_image_tensor(model, input_tensor, class_names):
    """
    Realiza la predicción de una imagen ya transformada y en tensor.

    Args:
        model: Modelo de PyTorch ya cargado y en eval mode.
        input_tensor: Tensor (1, C, H, W), ya preprocesado y en el dispositivo correcto.
        class_names: Lista con los nombres de las clases.

    Returns:
        Tuple (índice predicho, nombre de la clase)
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_index = predicted.item()
        predicted_class = class_names[predicted_index]
        return predicted_index, predicted_class
