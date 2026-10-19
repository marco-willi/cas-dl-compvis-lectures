# Augmentation Stability Demo

Interactive demo for the classification lecture (section "Data Augmentation" in
`pages/lectures/classification.qmd`): manipulate an image (flip, zoom, rotation, shift,
brightness, blur, occlusion) and compare the predictions of an ImageNet-pretrained
EfficientNet-B0 on the original and the manipulated image. The original image's top class
stays visible on the manipulated image, also when it leaves the top five.

Everything runs in the browser with ONNX Runtime Web (WebAssembly). There is no backend, and
all assets are loaded with relative paths from this folder, so the demo works under any
site sub-path such as `https://<user>.github.io/<repository>/`.

## Files

| Path                         | Purpose                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `index.html`, `app.js`       | Page, controls, manipulation, results                                         |
| `preprocess.js`              | The model's preprocessing (Pillow-compatible bicubic resize, crop, normalize) |
| `worker.js`                  | Web Worker that owns the single inference session                             |
| `model/efficientnet_b0.onnx` | Exported network (21.7 MB)                                                    |
| `model/meta.json`            | Class labels and preprocessing parameters of the weights                      |
| `model/reference.json`       | Python results used by the self-check                                         |
| `samples/`                   | Sample photographs (PNG) and `samples.json`                                   |
| `vendor/ort/`                | ONNX Runtime Web 1.30.0 (MIT license), 14 MB                                  |
| `tools/`                     | Scripts that generate `samples/` and `model/`                                 |

## Run locally

The demo has to be served over HTTP; it does not work from a `file://` page.

```bash
quarto preview            # then open /demos/cnns/augmentation_stability/
# or, without Quarto:
python -m http.server 8000 --directory demos/cnns/augmentation_stability
```

## Regenerate the assets

Only needed when changing the sample images or the model. Run from the repository root:

```bash
# 1. Sample images (resized copies of course images, written as lossless PNG)
uv run python demos/cnns/augmentation_stability/tools/prepare_samples.py

# 2. Model, labels and Python references (the onnx packages are only needed here)
uv run --with onnx --with onnxscript --with onnxruntime \
    python demos/cnns/augmentation_stability/tools/export_model.py
```

`export_model.py` exports `torchvision.models.efficientnet_b0` with
`EfficientNet_B0_Weights.IMAGENET1K_V1` in evaluation mode, checks ONNX Runtime against
PyTorch, and stores reference predictions. Run step 1 before step 2, because the references
are computed on the sample images.

To update ONNX Runtime Web, replace the three files in `vendor/ort/` with the ones from
`https://cdn.jsdelivr.net/npm/onnxruntime-web@<version>/dist/`.

## Verify browser against Python

Open the demo with `?selftest` appended to the URL, or expand "Verification against Python"
and press "Run self-check". It compares

1. the network alone, on a fixed input tensor that both sides generate identically, and
2. the full path (image decoding, preprocessing, network) on every sample image

with the PyTorch / torchvision results in `model/reference.json`.

## Embed in Quarto

HTML page (path relative to `pages/lectures/`):

````markdown
```{=html}
<iframe name="augmentation-stability" loading="lazy" data-auto-height
        src="../../demos/cnns/augmentation_stability/index.html"
        width="100%" height="820" style="border: 1px solid #ddd; border-radius: 6px;"
        title="Interactive demo: augmentation stability of a pretrained CNN"></iframe>
```
````

Reveal.js slide (path relative to `pages/slides/`; `data-src` loads the demo only when the
slide is near):

````markdown
## Augmentation Stability

```{=html}
<iframe name="augmentation-stability" data-src="../../demos/cnns/augmentation_stability/index.html"
        width="100%" height="850" style="border: 0;"
        title="Interactive demo: augmentation stability of a pretrained CNN"></iframe>
```
````

The folder is listed under `project.resources` in `_quarto.yml`, which copies it into the
rendered site. Above about 900 px width the controls sit beside the images; below that they
move on top and the demo gets taller (about 820 px). With `data-auto-height` the iframe
follows the height of the demo; `height` is then only the initial value. Leave the attribute
out on slides, where the height should stay fixed.

## How it works

- **Input image.** Uploads larger than 320 px (shorter side) or 640 px (longer side) are
  scaled down once. Original and manipulated image both start from this image.
- **Manipulation order.** Flip → zoom → rotation → shift are combined into one resampling
  step; then brightness, blur and occlusion are applied to the pixels. Positive angles rotate
  counter-clockwise. Areas outside the image are filled with black; brightness values above
  255 are clipped, and the demo reports the share of clipped pixels.
- **Preprocessing.** Both images then pass through the preprocessing of the weights: resize
  the shorter side to 256 (bicubic, antialiased), center crop 224, scale to [0, 1], normalize
  with the ImageNet mean and standard deviation. The dashed frame marks the center crop, and
  the "Model input" panels show exactly the pixels the network receives.
- **Inference.** The model is downloaded once and kept in one session inside a Web Worker.
  Previews update immediately; inference starts 120 ms after the last change, and results
  that are no longer current are discarded.
