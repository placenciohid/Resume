# Dynamic Zoom — contribution notes

## Project and contribution

Dynamic Zoom is a collaborative video super-resolution project. My contribution concerned model execution and displaying upscaled frames alongside the original stream. The original project implementation lives in the [upstream Dynamic-Zoom repository](https://github.com/Dynamic-Zoom/Dynamic-Zoom); this page is the portfolio entry and technical guide, not a mirror or a claim of sole authorship.

## Pipeline

`frame capture → region selection → preprocessing → GPU inference → resize/display → output`

Separate capture, inference, and rendering so a slow stage can be measured without confusing it with the others. Track source frame identifiers through the pipeline to compare corresponding input/output frames. Bound buffering to avoid a fast capture loop building an ever-growing queue.

### Integration sketch

```python
# Interface sketch, not executable upstream code.
for frame_id, frame in capture:
    tensor = preprocess(frame, region_of_interest)
    enhanced = model_executor(tensor)
    render_side_by_side(frame_id, frame, enhanced)
```

## Evaluating performance

Measure capture, preprocessing, model, and display separately. GPU measurements need synchronization and warm-up; report the input dimensions, output dimensions, device, precision, and batch size. The original website article reports historical latency, but this repository contains neither the hardware benchmark fixture nor weights needed to reproduce it. Do not read that figure as a newly validated end-to-end frame rate.

## Source and reproduction

Use the upstream README for its installation instructions and weight availability. Upstream code remains under its own terms. This entry links the contribution to the same Resume portfolio index as the other projects without copying third-party code or inventing missing model weights.

[Website article](https://placenciohid.github.io/2024/10/08/DynamicZoom.html)
