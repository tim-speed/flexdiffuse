# Composition Experimentation

Composition is accessble through the second tab in the UI.
Right now it functions as a poor proof of concept, but shows
    how guidance can be applied to a certain area.

## Why?

A compositional meta language or structure is essential for repeatable template driven asset generation...
Say you want to generate unique characters portraits for an entire game.. You would want to define a general artistic style, a base prompt, and then a modifier prompt or prompts that further tune aspects of the character.. a simple example could be face.

## What?

The current implmentation here takes multiple prompts, and then blends their noise guidance within a certain space.
What is interesting is that the "background" or main prompt, will adapt the space and noise around it to match the targeted guidance space.
A simple example is to place a bear in the middle of the image, and provide no background prompt. You will see that it guides the background around the bear, to match the heavily guided bear area in the middle.

## How?

See: `composition/guide.py`