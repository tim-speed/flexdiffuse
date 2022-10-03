'''Sandbox UI for testing all features'''
from typing import Callable
import gradio as gr

import utils


def block(runner: Callable[[], utils.Runner]):
    def run(prompt, init_image, guide_image, guide_text, height, width,
            mapping_concepts, guide_image_threshold_mult,
            guide_image_threshold_floor, guide_image_clustered,
            guide_image_linear_start, guide_image_linear_end,
            guide_image_max_guidance, guide_image_header, guide_image_mode,
            guide_image_reuse, strength, steps, guidance_scale, samples, seed,
            debug):
        if debug and samples * steps > 100:
            samples = 100 // steps
            print(f'Debug detected, forcing samples to {samples}'
                  f', to avoid too much output... ( <= 100 imgs )')
        if not guide_image and guide_text:
            guide = guide_text
        else:
            guide = guide_image
        imgs, grid = runner().gen(
            prompt, init_image, guide, (height, width), mapping_concepts,
            guide_image_threshold_mult, guide_image_threshold_floor,
            guide_image_clustered,
            (guide_image_linear_start, guide_image_linear_end),
            guide_image_max_guidance, guide_image_header, guide_image_mode,
            guide_image_reuse, strength, steps, guidance_scale, samples, seed,
            debug)
        return imgs

    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                prompt = gr.TextArea(
                    label='Enter your prompt',
                    show_label=False,
                    max_lines=1,
                    placeholder='Enter your prompt',
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                )
                generate: gr.Button = gr.Button(value='Generate image',
                                                variant='primary').style(
                                                    margin=False,
                                                    rounded=(False, True, True,
                                                             False),
                                                ) # type: ignore
            with gr.Row().style(equal_height=True):
                init_image = gr.Image(label='Initial image',
                                      source='upload',
                                      interactive=True,
                                      type='pil')
                with gr.Tab('Guide Image'):
                    guide_image = gr.Image(label='Guidance image',
                                           source='upload',
                                           interactive=True,
                                           type='pil')
                with gr.Tab('Guide Text'):
                    guide_text = gr.TextArea(
                        label='Text Guide ( only usable if no image is set )',
                        max_lines=1,
                        placeholder=('Enter text to blend with the prompt '
                                     'using the same techniques used for '
                                     'image mapping'),
                    )
        with gr.Row(equal_height=True):
            strength = gr.Slider(label='Diffusion Strength ( For Img2Img )',
                                 minimum=0,
                                 maximum=1,
                                 value=0.6,
                                 step=0.01)
            mapping_concepts = gr.TextArea(
                label='Image Guidance Mapping Concepts',
                max_lines=1,
                placeholder=
                'Enter items in image you would like to map directly',
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                steps = gr.Slider(label='Steps',
                                  minimum=8,
                                  maximum=100,
                                  value=30,
                                  step=2)
            with gr.Column(scale=1, variant='panel'):
                guide_image_threshold_mult = gr.Slider(
                    label='Threshold "Match" Guidance Multiplier ( Image )',
                    minimum=-1,
                    maximum=1,
                    value=0.25,
                    step=0.01)
            with gr.Column(scale=1, variant='panel'):
                guide_image_threshold_floor = gr.Slider(
                    label='Threshold "Match" Guidance Floor ( Image )',
                    minimum=0,
                    maximum=1,
                    value=0.75,
                    step=0.01)

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                samples = gr.Slider(label='Batches ( Images )',
                                    minimum=1,
                                    maximum=16,
                                    value=4,
                                    step=1)
            with gr.Column(scale=1, variant='panel'):
                guide_image_linear_start = gr.Slider(
                    label='Linear Guidance Start ( Image )',
                    minimum=-1,
                    maximum=1,
                    value=0.1,
                    step=0.01)
            with gr.Column(scale=1, variant='panel'):
                guide_image_linear_end = gr.Slider(
                    label='Linear Guidance End ( Image )',
                    minimum=-1,
                    maximum=1,
                    value=0.5,
                    step=0.01)

        with gr.Row(equal_height=True):
            guidance_scale = gr.Slider(label='Guidance Scale ( Overall )',
                                       minimum=0,
                                       maximum=20,
                                       value=8,
                                       step=0.5)
            guide_image_clustered = gr.Slider(
                label='Clustered "Match" Guidance ( Image )',
                minimum=-0.5,
                maximum=0.5,
                value=0.15,
                step=0.01)

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                seed = gr.Number(label='Seed',
                                 precision=0,
                                 value=0,
                                 interactive=True)
            with gr.Column(scale=1, variant='panel'):
                guide_image_max = gr.Slider(label='Max Image Guidance',
                                            minimum=0,
                                            maximum=1,
                                            value=0.35,
                                            step=0.01,
                                            interactive=True)
            with gr.Column(scale=1, variant='panel'):
                guide_image_header = gr.Slider(label='Max Image Header Mult',
                                               minimum=0,
                                               maximum=1,
                                               value=0,
                                               step=0.01,
                                               interactive=True)

        with gr.Row(equal_height=True):
            height = gr.Slider(minimum=64,
                               maximum=2048,
                               step=64,
                               label="Init Height",
                               value=512)
            width = gr.Slider(minimum=64,
                              maximum=2048,
                              step=64,
                              label="Init Width",
                              value=512)
            guide_image_mode = gr.Radio(label='Mapping Priority',
                                        choices=['Text Order', 'Optimal Fit'],
                                        value='Optimal Fit',
                                        type='index')
            with gr.Group(elem_id='cbgroup'):
                guide_image_reuse = gr.Checkbox(label='Reuse Latents',
                                                value=True)
                debug = gr.Checkbox(label='Export Debug Images', value=False)

        gallery = gr.Gallery(label='Generated images',
                             show_label=False,
                             elem_id='gallery').style(grid=2, height='auto')

        prompt.submit(run,
                      inputs=[
                          prompt, init_image, guide_image, guide_text, height,
                          width, mapping_concepts, guide_image_threshold_mult,
                          guide_image_threshold_floor, guide_image_clustered,
                          guide_image_linear_start, guide_image_linear_end,
                          guide_image_max, guide_image_header, guide_image_mode,
                          guide_image_reuse, strength, steps, guidance_scale,
                          samples, seed, debug
                      ],
                      outputs=[gallery])
        generate.click(run,
                       inputs=[
                           prompt, init_image, guide_image, guide_text, height,
                           width, mapping_concepts, guide_image_threshold_mult,
                           guide_image_threshold_floor, guide_image_clustered,
                           guide_image_linear_start, guide_image_linear_end,
                           guide_image_max, guide_image_header,
                           guide_image_mode, guide_image_reuse, strength, steps,
                           guidance_scale, samples, seed, debug
                       ],
                       outputs=[gallery])
