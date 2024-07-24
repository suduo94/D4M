# Modify Diffusers Library

Step 1:
Modify Diffusers source code `diffusers/src/diffusers/pipelines/stable_diffusion/__init__.py` to import the customized pipelines.
```python
### Original code
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    ### ……
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    ### ……

### Modified code
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    ### ……
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_gen_latents import StableDiffusionGenLatentsPipeline
    from .pipeline_stable_diffusion_latents2img import StableDiffusionLatents2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    ### ……
```

Step 2:
Modify Diffusers source code `diffusers/src/diffusers/pipelines/__init__.py` to import the customized pipelines.
```python
### Original code
    from .stable_diffusion import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……
    )

### Modified code
    from .stable_diffusion import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionGenLatentsPipeline,
        StableDiffusionLatents2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……
    )
```
Step 3:
Modify Diffusers source code `diffusers/src/diffusers/__init__.py` to import the customized pipelines.
```python
### Original code
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……

    )

### Modified code
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionGenLatentsPipeline,
        StableDiffusionLatents2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……
    )
```