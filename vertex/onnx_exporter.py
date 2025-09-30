import torch
import logging
from torch.export import Dim

def export_model_to_onnx(model: torch.nn.Module, model_type: str, file_path: str, seq_len: int, input_size: int):
    model.eval().to("cpu")
    
    batch_dim = Dim("batch_size")
    seq_dim = Dim("seq_len")

    dummy_x = torch.randn(1, seq_len, input_size, device="cpu")
    args_tuple = (dummy_x,)
    
    if model_type == "transformer":
        input_names = ['src']
    else:
        input_names = ['x']
        
    dynamic_shapes = {input_names[0]: {0: batch_dim, 1: seq_dim}}
    output_names = ['output']
    
    if model_type == "lstm_attention":
        num_directions = 2 if model_type == "lstm_attention" else 1
        h0 = torch.randn(model.num_layers * num_directions, 1, model.hidden_size, device="cpu")
        c0 = torch.randn(model.num_layers * num_directions, 1, model.hidden_size, device="cpu")
        args_tuple = (dummy_x, (h0, c0))
        input_names.append('hidden')
        dynamic_shapes["hidden"] = ({1: batch_dim}, {1: batch_dim})
        logging.info(f"'{model_type}' requires hidden states for export.")

    logging.info(f"Exporting '{model_type}' to ONNX using dynamo: {file_path}")
    
    try:
        no_cf = (model_type in ["tcn", "patchtst"])
        if model_type in ["transformer", "patchtst", "tcn"]:
            dynamic_axes = {
                input_names[0]: {0: 'batch_size', 1: 'seq_len'},
                'output': {0: 'batch_size'}
            }
            torch.onnx.export(
                model,
                args_tuple,
                file_path,
                opset_version=17,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=not no_cf,
                dynamo=False,
            )
        else:
            torch.onnx.export(
                model,
                args_tuple,
                file_path,
                opset_version=20,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                do_constant_folding=True,
                dynamo=True,
            )
        logging.info(f"Successfully exported '{model_type}' to ONNX: {file_path}")
    except Exception as e:
        logging.error(f"Failed to export '{model_type}' to ONNX. Error: {e}", exc_info=True)