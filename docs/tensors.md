[← Tora](../README.md)

# Tensor inspection and examples

Run commands from the repository root. See [module capture](capture.md) for
a complete example with external weights and a file-backed input.

The PyTorch tensor provider implements `ITensor::ReadRegion` with `IArray`
origin and shape inputs containing one Int64 per dimension. Region reads return
independent, contiguous CPU tensors with the original dtype and rank. The editor
formats their values through read-only host access; it displays pages of up to
256 elements and copies only requested row regions. Device reads require clicking
Read values. Dense, non-quantized tensors with storage are supported, including
non-contiguous views. Editing values is not yet implemented.

Rebuild the SDK consumers, Node Editor, and extension together after this
`ITensor` ABI change; binaries implementing the older interface are incompatible.
Callers must prevent concurrent mutation during a region read and retain the
returned tensor while using its host pointer.

Run `scripts\test-tensor-region.cmd` to check native CPU region reads with
libtorch. It checks coordinates, independent storage, dtypes, views, scalars,
empty regions, and invalid inputs. It does not test window behavior or GPU
transfers in the editor.

Run `scripts\test-tensor-region-abi.cmd` with the updated local SDK to check the
`IArray` ABI boundary, malformed inputs, snapshot ownership, and host access.

TensorEditor displays values on the left and a D3D heatmap panel on the right.
CPU slices load automatically; use **Read slice** for device tensors. The last two dimensions
are the image axes; enter indices for the leading dimensions (for example,
`0, 3` for batch 0, channel 3 of an NCHW tensor), or use Previous/Next. Reading
a device slice copies that region to CPU. Choose
Neon, Diverging, or Gray palettes, optional contours, and Auto/Locked ranges.
Hover shows the original typed value; colors use a floating-point display
representation. Magenta marks NaN or infinity. Real numeric and Boolean tensors
with at least two dimensions are supported, up to 4096 per image axis and
4,194,304 elements per slice. Complex tensors and empty image dimensions are
reported as unsupported.

For a larger convolution example, run from the repository root:

```powershell
_build/capture-venv/Scripts/python.exe examples/save_conv2d_features.py
npm run build
node examples/conv2d-features.mjs
```

Open `_build/conv2d-features.nodegraph.json` in the Node Editor. Its 18 nodes load
the input and weights from `_build/conv2d-features.safetensors`, then run four
convolution/ReLU stages: 5×5 feature filters, dilated depthwise convolution,
1×1 channel mixing, and stride-2 downsampling. The two-image input has shape
`[2, 3, 96, 128]`; the final output is `[2, 6, 48, 64]`. The graph also exposes
the input and each intermediate activation for inspection. Try slice indices
`0, 0`, `0, 3`, or `1, 5` on the final tensor, and lock the range to compare channels.
These are deterministic demonstration filters, not a trained model.

The configuration is in `examples/conv2d-features.json`. Generated graphs store
an absolute Base Directory for their relative tensor filename; regenerate the
graph or update that property if moving the files to another machine. Python
writes reference tensors and summary statistics separately. Run
`scripts\test-convolution-features.cmd` to compare all four native generated
convolution/ReLU stages against those references. This does not execute the graph
inside the Node Editor.
