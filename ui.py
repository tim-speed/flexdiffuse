'''Simple Test UI'''
import sys
import gradio as gr

import utils
import interface.sandbox as sandbox

runner = None


def get_runner() -> utils.Runner:
    global runner
    if runner is None:
        runner = utils.Runner(
            not [s for s in sys.argv[1:] if 'dl' in s or 'download' in s])
    return runner


def launch():
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
        with gr.Tab('Sandbox'):
            sandbox.block(get_runner)

    block.launch(debug=True)


if __name__ == '__main__':
    launch()