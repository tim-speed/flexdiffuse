'''Sandbox UI for testing all features'''
from typing import Any, Callable, Iterable, List
import gradio as gr

import utils
from composition.schema import EntitySchema, Schema

DEFAULT_SCHEMA = Schema('A forest', 'Photo', 'Painting', (0.0, 1.0), [
    EntitySchema('A bear', (16, 128), (256, 256)),
    EntitySchema('A deer', (288, 160), (192, 192))
])


def unpack(e: object) -> List[Any]:
    nout = []
    for v in e.__dict__.values():
        if not isinstance(v, str) and isinstance(v, Iterable):
            nout.extend(v)
        else:
            nout.append(v)
    return nout


def block(runner: Callable[[], utils.Runner]):
    def run(bg_prompt, entities_df, start_style, end_style,
            style_blend_linear_start, style_blend_linear_end, init_image,
            samples, strength, steps, guidance_scale, height, width, seed,
            debug):
        if debug and samples * steps > 100:
            samples = 100 // steps
            print(f'Debug detected, forcing samples to {samples}'
                  f', to avoid too much output... ( <= 100 imgs )')
        imgs, grid = runner().compose(
            bg_prompt, entities_df, start_style, end_style,
            (style_blend_linear_start, style_blend_linear_end), init_image,
            samples, strength, steps, guidance_scale, (height, width), seed,
            debug)
        return imgs

    with gr.Group():
        with gr.Box():
            with gr.Row():
                bg_prompt = gr.TextArea(
                    label='Background / Main Prompt',
                    value=DEFAULT_SCHEMA.background_prompt,
                    max_lines=1,
                    placeholder=(
                        'Enter text to define the main scene / background.'),
                )
            with gr.Row():
                entities_df = gr.Dataframe(
                    label='Entities ( Ordered )',
                    value=[unpack(e) for e in DEFAULT_SCHEMA.entities],
                    headers=[
                        'Prompt', 'Left', 'Top', 'Width', 'Height', 'Strength'
                    ],
                    datatype=[
                        'str', 'number', 'number', 'number', 'number', 'number'
                    ],
                    col_count=(6, 'fixed'),
                    interactive=True)
            with gr.Row():
                start_style = gr.TextArea(
                    label='Starting Style Prompt',
                    value=DEFAULT_SCHEMA.style_start_prompt,
                    max_lines=1,
                    placeholder=(
                        'Enter text to define the style early on in generation.'
                    ),
                )

            with gr.Row():
                end_style = gr.TextArea(
                    label='Ending Style Prompt',
                    value=DEFAULT_SCHEMA.style_end_prompt,
                    max_lines=1,
                    placeholder=(
                        'Enter text to define the final style in generation.'),
                )

            with gr.Row():
                style_blend_linear_start = gr.Slider(
                    label='Linear Style Blend Start',
                    minimum=-1,
                    maximum=1,
                    value=DEFAULT_SCHEMA.style_blend[0],
                    step=0.01)
                style_blend_linear_end = gr.Slider(
                    label='Linear Style Blend End',
                    minimum=-1,
                    maximum=1,
                    value=DEFAULT_SCHEMA.style_blend[1],
                    step=0.01)

            with gr.Row():
                init_image = gr.Image(label='Initial image',
                                      source='upload',
                                      interactive=True,
                                      type='pil')
            with gr.Row():
                samples = gr.Slider(label='Batches ( Images )',
                                    minimum=1,
                                    maximum=16,
                                    value=4,
                                    step=1)

                strength = gr.Slider(label='Diffusion Strength ( For Img2Img )',
                                     minimum=0,
                                     maximum=1,
                                     value=0.6,
                                     step=0.01)

            with gr.Row():
                steps = gr.Slider(label='Steps',
                                  minimum=8,
                                  maximum=100,
                                  value=30,
                                  step=2)
                guidance_scale = gr.Slider(label='Guidance Scale ( Overall )',
                                           minimum=0,
                                           maximum=20,
                                           value=8,
                                           step=0.5)

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

            with gr.Row():
                seed = gr.Number(label='Seed',
                                 precision=0,
                                 value=0,
                                 interactive=True)
                debug = gr.Checkbox(label='Export Debug Images', value=False)
                generate: gr.Button = gr.Button(value='Generate image',
                                                variant='primary')

        gallery = gr.Gallery(label='Generated images',
                             show_label=False,
                             elem_id='gallery').style(grid=2, height='auto')

        bg_prompt.submit(run,
                         inputs=[
                             bg_prompt, entities_df, start_style, end_style,
                             style_blend_linear_start, style_blend_linear_end,
                             init_image, samples, strength, steps,
                             guidance_scale, height, width, seed, debug
                         ],
                         outputs=[gallery])
        generate.click(run,
                       inputs=[
                           bg_prompt, entities_df, start_style, end_style,
                           style_blend_linear_start, style_blend_linear_end,
                           init_image, samples, strength, steps, guidance_scale,
                           height, width, seed, debug
                       ],
                       outputs=[gallery])
