'''Simple Test UI'''
import sys
from turtle import width
import gradio as gr

import utils

runner = utils.Runner(
    not [s for s in sys.argv[1:] if 'dl' in s or 'download' in s])


def run(prompt, init_image, guide_image, height, width, prompt_text_vs_image,
        guide_image_bg_fg, guide_image_mode, strength, steps, guidance_scale,
        samples, seed):
    imgs, grid = runner.gen(prompt, init_image, guide_image, (height, width),
                            prompt_text_vs_image, guide_image_bg_fg,
                            guide_image_mode, strength, steps, guidance_scale,
                            samples, seed)
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
'''
block = gr.Blocks(css=css)

with block:
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                prompt = gr.TextArea(
                    label='Enter your prompt',
                    show_label=False,
                    max_lines=1,
                    placeholder='Enter your prompt',
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                generate: gr.Button = gr.Button(value='Generate image',
                                                variant='primary').style(
                                                    margin=False,
                                                    rounded=(False, True, True,
                                                             False),
                                                ) # type: ignore
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                init_image = gr.Image(label='Initial image',
                                      source='upload',
                                      interactive=True,
                                      type='pil')
                guide_image = gr.Image(label='Guidance image',
                                       source='upload',
                                       interactive=True,
                                       type='pil')

        with gr.Row(mobile_collapse=False, equal_height=True):
            strength = gr.Slider(label='Diffusion Strength',
                                 minimum=0,
                                 maximum=1,
                                 value=0.6,
                                 step=0.01)
            prompt_text_vs_image = gr.Slider(label='Image Guidance from Text',
                                             minimum=0,
                                             maximum=1,
                                             value=0.25,
                                             step=0.01)

        with gr.Row():
            steps = gr.Slider(label='Steps',
                              minimum=8,
                              maximum=50,
                              value=30,
                              step=2)
            guide_image_bg_fg = gr.Slider(
                label='Image Style vs Subject Guidance',
                minimum=0,
                maximum=1,
                value=0.8,
                step=0.01)

        with gr.Row():
            samples = gr.Slider(label='Images',
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1)
            guide_image_mode = gr.Radio(label='Mapping Priority',
                                        choices=['Text Order', 'Optimal Fit'],
                                        value='Text Order',
                                        type='index')

        with gr.Row():
            guidance_scale = gr.Slider(label='Guidance Scale',
                                       minimum=0,
                                       maximum=20,
                                       value=8,
                                       step=0.5)
            seed = gr.Number(label='Seed',
                             precision=0,
                             value=0,
                             interactive=True)

        with gr.Group():
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
                          prompt_text_vs_image, guide_image_bg_fg,
                          guide_image_mode, strength, steps, guidance_scale,
                          samples, seed
                      ],
                      outputs=[gallery])
        generate.click(run,
                       inputs=[
                           prompt, init_image, guide_image, height, width,
                           prompt_text_vs_image, guide_image_bg_fg,
                           guide_image_mode, strength, steps, guidance_scale,
                           samples, seed
                       ],
                       outputs=[gallery])

block.launch(debug=True)