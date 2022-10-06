'''Simple Test UI'''
import sys
import gradio as gr

import utils
import interface.composer as composer
import interface.sandbox as sandbox

runner = None
pargs = [a.strip().lower() for a in sys.argv[1:]]


def _has_arg_like(*args: str) -> bool:
    return bool([pa for pa in pargs for a in args if a in pa])


def get_runner() -> utils.Runner:
    global runner
    if runner is None:
        runner = utils.Runner(not _has_arg_like('dl', 'download'))
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
        div.row>div.col>div, div.row>div.col>div>div, div.row>div.col fieldset,\
            #cbgroup, #cbgroup>div {
            min-height: 100%;
        }
        div#cbgroup {
            max-width: 25%
        }
        
    '''
    block = gr.Blocks(css=css)

    with block:
        with gr.Tab('Sandbox'):
            sandbox.block(get_runner)
        with gr.Tab('Compose'):
            composer.block(get_runner)

    block.launch(server_name=('0.0.0.0' if _has_arg_like('lan') else None),
                 debug=True)


if __name__ == '__main__':
    launch()