#!/usr/bin/env python3
"""
Generate numerical reference fixtures for Edifice forward pass validation.

Downloads real HuggingFace models, runs forward passes on deterministic inputs,
and saves inputs + expected outputs as .safetensors files.

Requirements:
    pip install torch transformers safetensors numpy

Usage:
    python scripts/generate_numerical_fixtures.py

Output:
    test/fixtures/numerical/vit_reference.safetensors      (~600KB)
    test/fixtures/numerical/whisper_encoder_reference.safetensors (~100KB)
"""

import os
import sys
import numpy as np
import torch
from safetensors.torch import save_file


def generate_vit_fixture():
    """Generate ViT forward pass reference.

    Model: google/vit-base-patch16-224
    Input: deterministic [1, 3, 224, 224] via torch.manual_seed(42)
    Output: logits [1, 1000]
    """
    from transformers import ViTForImageClassification

    print("Loading google/vit-base-patch16-224...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.eval()

    # Deterministic input
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224)

    print("Running ViT forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 1000]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "vit_reference.safetensors"
    )

    save_file({
        "input": pixel_values,
        "expected_logits": logits,
    }, out_path)

    print(f"Saved ViT fixture to {out_path}")
    print(f"  Input shape: {pixel_values.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, :5].tolist()}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


def generate_whisper_encoder_fixture():
    """Generate Whisper encoder forward pass reference.

    Model: openai/whisper-base
    Input: deterministic mel spectrogram [1, 80, 100] via torch.manual_seed(42)
    Output: encoder hidden states [1, 50, 512]
        (100 mel frames -> 50 after stride-2 convolution)
    """
    from transformers import WhisperModel

    print("\nLoading openai/whisper-base...")
    model = WhisperModel.from_pretrained("openai/whisper-base")
    model.eval()

    # Deterministic mel input (shorter than full 3000 frames for fixture size)
    torch.manual_seed(42)
    mel_input = torch.randn(1, 80, 100)

    print("Running Whisper encoder forward pass...")
    with torch.no_grad():
        # Run encoder only
        encoder_output = model.encoder(mel_input)
        hidden_states = encoder_output.last_hidden_state  # [1, 50, 512]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "whisper_encoder_reference.safetensors"
    )

    save_file({
        "mel_input": mel_input,
        "expected_encoder_output": hidden_states,
    }, out_path)

    print(f"Saved Whisper encoder fixture to {out_path}")
    print(f"  Mel input shape: {mel_input.shape}")
    print(f"  Encoder output shape: {hidden_states.shape}")
    print(f"  Output sample: {hidden_states[0, 0, :5].tolist()}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


def generate_convnext_fixture():
    """Generate ConvNeXt forward pass reference.

    Model: facebook/convnext-tiny-224
    Input: deterministic [1, 3, 224, 224] via torch.manual_seed(42)
    Output: logits [1, 1000]
    """
    from transformers import ConvNextForImageClassification

    print("\nLoading facebook/convnext-tiny-224...")
    model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
    model.eval()

    # Deterministic input
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224)

    print("Running ConvNeXt forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 1000]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "convnext_reference.safetensors"
    )

    save_file({
        "input": pixel_values,
        "expected_logits": logits,
    }, out_path)

    print(f"Saved ConvNeXt fixture to {out_path}")
    print(f"  Input shape: {pixel_values.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, :5].tolist()}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


def generate_resnet_fixture():
    """Generate ResNet-50 forward pass reference.

    Model: microsoft/resnet-50
    Input: deterministic [1, 3, 224, 224] via torch.manual_seed(42)
    Output: logits [1, 1000]

    HuggingFace ResNet-50 state_dict key structure:
      Stem:
        resnet.embedder.embedder.convolution.weight        # Conv2d(3, 64, 7x7) no bias
        resnet.embedder.embedder.normalization.{weight,bias,running_mean,running_var}
      Bottleneck blocks:
        resnet.encoder.stages.{s}.layers.{l}.layer.{k}.convolution.weight
        resnet.encoder.stages.{s}.layers.{l}.layer.{k}.normalization.{weight,bias,...}
        k=0: 1x1 reduce, k=1: 3x3, k=2: 1x1 expand
      Shortcut (projection, layer 0 of each stage):
        resnet.encoder.stages.{s}.layers.0.shortcut.convolution.weight
        resnet.encoder.stages.{s}.layers.0.shortcut.normalization.{weight,bias,...}
      Classifier:
        classifier.1.weight     # nn.Linear(2048, 1000)
        classifier.1.bias
    """
    from transformers import ResNetForImageClassification

    print("\nLoading microsoft/resnet-50...")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()

    # Deterministic input
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224)

    print("Running ResNet-50 forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 1000]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "resnet_reference.safetensors"
    )

    save_file({
        "input": pixel_values,
        "expected_logits": logits,
    }, out_path)

    print(f"Saved ResNet-50 fixture to {out_path}")
    print(f"  Input shape: {pixel_values.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, :5].tolist()}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


def generate_detr_fixture():
    """Generate DETR forward pass reference.

    Model: facebook/detr-resnet-50
    Input: deterministic [1, 3, 800, 800] via torch.manual_seed(42)
    Output: class_logits [1, 100, 92], bbox_pred [1, 100, 4]

    Note: DETR uses 800x800 default image size and outputs 92 classes
    (91 COCO classes + 1 no-object class).
    """
    from transformers import DetrForObjectDetection

    print("\nLoading facebook/detr-resnet-50...")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()

    # Deterministic input (smaller than default 800x800 for fixture size)
    # Use 256x256 to keep fixture manageable while still testing the full pipeline
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 256, 256)

    print("Running DETR forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        class_logits = outputs.logits        # [1, 100, 92]
        bbox_pred = outputs.pred_boxes       # [1, 100, 4]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "detr_reference.safetensors"
    )

    save_file({
        "input": pixel_values,
        "expected_class_logits": class_logits,
        "expected_bbox_pred": bbox_pred,
    }, out_path)

    print(f"Saved DETR fixture to {out_path}")
    print(f"  Input shape: {pixel_values.shape}")
    print(f"  Class logits shape: {class_logits.shape}")
    print(f"  BBox pred shape: {bbox_pred.shape}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    generate_vit_fixture()
    generate_whisper_encoder_fixture()
    generate_convnext_fixture()
    generate_resnet_fixture()
    generate_detr_fixture()
    print("\nAll fixtures generated successfully!")
