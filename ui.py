'''Simple Test UI'''
import sys
from turtle import width
import gradio as gr

import utils

runner = None


def run(prompt, init_image, guide_image, height, width,
        guide_image_threshold_mult, guide_image_threshold_floor,
        guide_image_clustered, guide_image_linear, guide_image_max_guidance,
        guide_image_mode, guide_image_reuse, strength, steps, guidance_scale,
        samples, seed):
    global runner
    if runner is None:
        runner = utils.Runner(
            not [s for s in sys.argv[1:] if 'dl' in s or 'download' in s])
    imgs, grid = runner.gen(prompt, init_image, guide_image, (height, width),
                            guide_image_threshold_mult,
                            guide_image_threshold_floor, guide_image_clustered,
                            guide_image_linear, guide_image_max_guidance,
                            guide_image_mode, guide_image_reuse, strength,
                            steps, guidance_scale, samples, seed)
    return imgs


css = '''
    textarea {
        max-height: 60px;
    }
    div.gr-block button.gr-button {
        max-width: 200px;
    }
    #gallery>div>.h-full {
        min-height: 20rem;
    }
    div.row, div.row>div.col {
        gap: 0;
        padding: 0;
    }
    div.row>div.col>div, div.row>div.col>div>div, div.row>div.col fieldset {
        min-height: 100%;
    }
'''
block = gr.Blocks(css=css)

with block:
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
                guide_image = gr.Image(label='Guidance image',
                                       source='upload',
                                       interactive=True,
                                       type='pil')

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                strength = gr.Slider(label='Diffusion Strength ( For Img2Img )',
                                     minimum=0,
                                     maximum=1,
                                     value=0.6,
                                     step=0.01)
            with gr.Column(scale=1, variant='panel'):
                guide_image_threshold_mult = gr.Slider(
                    label='Threshold "Match" Guidance Multiplier ( Image )',
                    minimum=-1,
                    maximum=1,
                    value=0.25,
                    step=0.05)
            with gr.Column(scale=1, variant='panel'):
                guide_image_threshold_floor = gr.Slider(
                    label='Threshold "Match" Guidance Floor ( Image )',
                    minimum=0,
                    maximum=1,
                    value=0.75,
                    step=0.05)

        with gr.Row(equal_height=True):
            steps = gr.Slider(label='Steps',
                              minimum=8,
                              maximum=100,
                              value=30,
                              step=2)
            guide_image_clustered = gr.Slider(
                label='Clustered "Match" Guidance ( Image )',
                minimum=-0.5,
                maximum=0.5,
                value=0.15,
                step=0.05)

        with gr.Row(equal_height=True):
            samples = gr.Slider(label='Batches ( Images )',
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1)
            guide_image_linear = gr.Slider(
                label='Linear "Style" Guidance ( Image )',
                minimum=-1,
                maximum=1,
                value=0.5,
                step=0.05)

        with gr.Row(equal_height=True):
            guidance_scale = gr.Slider(label='Guidance Scale ( Overall )',
                                       minimum=0,
                                       maximum=20,
                                       value=8,
                                       step=0.5)
            guide_image_max = gr.Slider(label='Max Image Guidance',
                                        minimum=0,
                                        maximum=1,
                                        value=0.35,
                                        step=0.05,
                                        interactive=True)

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                seed = gr.Number(label='Seed',
                                 precision=0,
                                 value=0,
                                 interactive=True)
            with gr.Column(scale=1, variant='panel'):
                guide_image_mode = gr.Radio(
                    label='Mapping Priority',
                    choices=['Text Order', 'Optimal Fit'],
                    value='Optimal Fit',
                    type='index')
            with gr.Column(scale=1, variant='panel'):
                guide_image_reuse = gr.Checkbox(label='Reuse Latents',
                                                value=True)

        with gr.Row():
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

        gallery = gr.Gallery(label='Generated images',
                             show_label=False,
                             elem_id='gallery').style(grid=2, height='auto')

        prompt.submit(run,
                      inputs=[
                          prompt, init_image, guide_image, height, width,
                          guide_image_threshold_mult,
                          guide_image_threshold_floor, guide_image_clustered,
                          guide_image_linear, guide_image_max, guide_image_mode,
                          guide_image_reuse, strength, steps, guidance_scale,
                          samples, seed
                      ],
                      outputs=[gallery])
        generate.click(run,
                       inputs=[
                           prompt, init_image, guide_image, height, width,
                           guide_image_threshold_mult,
                           guide_image_threshold_floor, guide_image_clustered,
                           guide_image_linear, guide_image_max,
                           guide_image_mode, guide_image_reuse, strength, steps,
                           guidance_scale, samples, seed
                       ],
                       outputs=[gallery])

block.launch(debug=True)