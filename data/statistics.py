"""
These codes are from https://github.com/tctianchi/pyvenn/blob/master/venn.py

"""
import json
import os
# coding: utf-8
from itertools import chain

from get_d4j_bug_list import get_history_defects4j_project_and_bug

try:
    # since python 3.10
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib import colors
# import math


def read_manual_d4j12(file="./manual_mcts_gpt_3.5_turbo_32patch.txt"):
    projects = ["Chart", "Time", "Mockito", "Lang", "Math", "Closure"]
    sets = set()
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            if line.startswith("-------"):
                continue
            if line.strip().split(",")[1] != "True":
                continue
            if line.split(",")[0].split("_")[0] not in projects:
                continue
            if line.split(",")[0] in sets:
                print(line.split(",")[0])
            sets.add(line.split(",")[0])
    return sets


def read_manual_d4j2(file="./manual_mcts_gpt_3.5_turbo_32patch.txt"):
    projects = ["Chart", "Time", "Mockito", "Lang", "Math", "Closure"]
    sets = set()
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            if line.startswith("-------"):
                continue
            if line.split(",")[1] != "True":
                continue
            if line.split(",")[0].split("_")[0] in projects:
                continue
            if line.split(",")[0] in sets:
                print(line.split(",")[0])
            sets.add(line.split(",")[0])
    return sets


default_colors = [
    # r, g, b, a
    [92, 192, 98, 0.5],
    [90, 155, 212, 0.5],
    [246, 236, 86, 0.6],
    [241, 90, 96, 0.4],
    [255, 117, 0, 0.3],
    [82, 82, 190, 0.2],
]
default_colors = [
    [i[0] / 255.0, i[1] / 255.0, i[2] / 255.0, i[3]]
    for i in default_colors
]


def draw_ellipse(fig, ax, x, y, w, h, a, fillcolor):
    e = patches.Ellipse(
        xy=(x, y),
        width=w,
        height=h,
        angle=a,
        color=fillcolor)
    ax.add_patch(e)


def draw_triangle(fig, ax, x1, y1, x2, y2, x3, y3, fillcolor):
    xy = [
        (x1, y1),
        (x2, y2),
        (x3, y3),
    ]
    polygon = patches.Polygon(
        xy=xy,
        closed=True,
        color=fillcolor)
    ax.add_patch(polygon)


def draw_text(fig, ax, x, y, text, color=[0, 0, 0, 1], fontsize=14, ha="center", va="center"):
    ax.text(
        x, y, text,
        horizontalalignment=ha,
        verticalalignment=va,
        fontsize=fontsize,
        color="black")


def draw_annotate(fig, ax, x, y, textx, texty, text, color=[0, 0, 0, 1], arrowcolor=[0, 0, 0, 0.3]):
    plt.annotate(
        text,
        xy=(x, y),
        xytext=(textx, texty),
        arrowprops=dict(color=arrowcolor, shrink=0, width=0.5, headwidth=8),
        fontsize=14,
        color=color,
        xycoords="data",
        textcoords="data",
        horizontalalignment='center',
        verticalalignment='center'
    )


def get_labels(data, fill=["number"]):
    """
    get a dict of labels for groups in data

    @type data: list[Iterable]
    @rtype: dict[str, str]

    input
      data: data to get label for
      fill: ["number"|"logic"|"percent"]

    return
      labels: a dict of labels for different sets

    example:
    In [12]: get_labels([range(10), range(5,15), range(3,8)], fill=["number"])
    Out[12]:
    {'001': '0',
     '010': '5',
     '011': '0',
     '100': '3',
     '101': '2',
     '110': '2',
     '111': '3'}
    """

    N = len(data)

    sets_data = [set(data[i]) for i in range(N)]  # sets for separate groups
    s_all = set(chain(*data))  # union of all sets

    # bin(3) --> '0b11', so bin(3).split('0b')[-1] will remove "0b"
    set_collections = {}
    for n in range(1, 2 ** N):
        key = bin(n).split('0b')[-1].zfill(N)
        value = s_all
        sets_for_intersection = [sets_data[i] for i in range(N) if key[i] == '1']
        sets_for_difference = [sets_data[i] for i in range(N) if key[i] == '0']
        for s in sets_for_intersection:
            value = value & s
        for s in sets_for_difference:
            value = value - s
        set_collections[key] = value

    labels = {k: "" for k in set_collections}
    if "logic" in fill:
        for k in set_collections:
            labels[k] = k + ": "
    if "number" in fill:
        for k in set_collections:
            labels[k] += str(len(set_collections[k]))
    if "percent" in fill:
        data_size = len(s_all)
        for k in set_collections:
            labels[k] += "(%.1f%%)" % (100.0 * len(set_collections[k]) / data_size)

    return labels


def venn2(labels, names=['A', 'B'], **options):
    """
    plots a 2-set Venn diagram

    @type labels: dict[str, str]
    @type names: list[str]
    @rtype: (Figure, AxesSubplot)

    input
      labels: a label dict where keys are identified via binary codes ('01', '10', '11'),
              hence a valid set could look like: {'01': 'text 1', '10': 'text 2', '11': 'text 3'}.
              unmentioned codes are considered as ''.
      names:  group names
      more:   colors, figsize, dpi, fontsize

    return
      pyplot Figure and AxesSubplot object
    """
    colors = options.get('colors', [default_colors[i] for i in range(2)])
    figsize = options.get('figsize', (9, 7))
    dpi = options.get('dpi', 96)
    fontsize = options.get('fontsize', 14)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=0.7)
    ax.set_xlim(left=0.0, right=1.0)

    # body
    draw_ellipse(fig, ax, 0.375, 0.3, 0.5, 0.5, 0.0, colors[0])
    draw_ellipse(fig, ax, 0.625, 0.3, 0.5, 0.5, 0.0, colors[1])
    draw_text(fig, ax, 0.74, 0.30, labels.get('01', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.26, 0.30, labels.get('10', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.30, labels.get('11', ''), fontsize=fontsize)

    # legend
    draw_text(fig, ax, 0.20, 0.56, names[0], colors[0], fontsize=fontsize, ha="right", va="bottom")
    draw_text(fig, ax, 0.80, 0.56, names[1], colors[1], fontsize=fontsize, ha="left", va="bottom")
    leg = ax.legend(names, loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True)
    leg.get_frame().set_alpha(0.5)

    return fig, ax


def venn3(labels, names=['A', 'B', 'C'], **options):
    """
    plots a 3-set Venn diagram

    @type labels: dict[str, str]
    @type names: list[str]
    @rtype: (Figure, AxesSubplot)

    input
      labels: a label dict where keys are identified via binary codes ('001', '010', '100', ...),
              hence a valid set could look like: {'001': 'text 1', '010': 'text 2', '100': 'text 3', ...}.
              unmentioned codes are considered as ''.
      names:  group names
      more:   colors, figsize, dpi, fontsize

    return
      pyplot Figure and AxesSubplot object
    """
    colors = options.get('colors', [default_colors[i] for i in range(3)])
    figsize = options.get('figsize', (9, 9))
    dpi = options.get('dpi', 96)
    fontsize = options.get('fontsize', 25)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(left=0.0, right=1.0)

    # body
    draw_ellipse(fig, ax, 0.333, 0.633, 0.5, 0.5, 0.0, colors[0])
    draw_ellipse(fig, ax, 0.666, 0.633, 0.5, 0.5, 0.0, colors[1])
    draw_ellipse(fig, ax, 0.500, 0.310, 0.5, 0.5, 0.0, colors[2])
    draw_text(fig, ax, 0.50, 0.27, labels.get('001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.73, 0.65, labels.get('010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.61, 0.46, labels.get('011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.27, 0.65, labels.get('100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.39, 0.46, labels.get('101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.65, labels.get('110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.51, labels.get('111', ''), fontsize=fontsize)

    # legend
    draw_text(fig, ax, 0.15, 0.87, names[0], colors[0], fontsize=fontsize, ha="right", va="bottom")
    draw_text(fig, ax, 0.85, 0.87, names[1], colors[1], fontsize=fontsize, ha="left", va="bottom")
    draw_text(fig, ax, 0.50, 0.02, names[2], colors[2], fontsize=fontsize, va="top")
    # leg = ax.legend(names, loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    plt.savefig(options.get("output", "./pdf/venn3.pdf"), bbox_inches='tight')

    return fig, ax


def venn4(labels, names=['A', 'B', 'C', 'D'], **options):
    """
    plots a 4-set Venn diagram

    @type labels: dict[str, str]
    @type names: list[str]
    @rtype: (Figure, AxesSubplot)

    input
      labels: a label dict where keys are identified via binary codes ('0001', '0010', '0100', ...),
              hence a valid set could look like: {'0001': 'text 1', '0010': 'text 2', '0100': 'text 3', ...}.
              unmentioned codes are considered as ''.
      names:  group names
      more:   colors, figsize, dpi, fontsize

    return
      pyplot Figure and AxesSubplot object
    """
    colors = options.get('colors', [default_colors[i] for i in range(4)])
    figsize = options.get('figsize', (12, 12))
    dpi = options.get('dpi', 96)
    fontsize = options.get('fontsize', 14)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(left=0.0, right=1.0)

    # body
    draw_ellipse(fig, ax, 0.350, 0.400, 0.72, 0.45, 140.0, colors[0])
    draw_ellipse(fig, ax, 0.450, 0.500, 0.72, 0.45, 140.0, colors[1])
    draw_ellipse(fig, ax, 0.544, 0.500, 0.72, 0.45, 40.0, colors[2])
    draw_ellipse(fig, ax, 0.644, 0.400, 0.72, 0.45, 40.0, colors[3])
    draw_text(fig, ax, 0.85, 0.42, labels.get('0001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.68, 0.72, labels.get('0010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.77, 0.59, labels.get('0011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.32, 0.72, labels.get('0100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.71, 0.30, labels.get('0101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.66, labels.get('0110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.65, 0.50, labels.get('0111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.14, 0.42, labels.get('1000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.17, labels.get('1001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.29, 0.30, labels.get('1010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.39, 0.24, labels.get('1011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.23, 0.59, labels.get('1100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.61, 0.24, labels.get('1101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.35, 0.50, labels.get('1110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.38, labels.get('1111', ''), fontsize=fontsize)

    # legend
    draw_text(fig, ax, 0.13, 0.18, names[0], colors[0], fontsize=fontsize, ha="right")
    draw_text(fig, ax, 0.18, 0.83, names[1], colors[1], fontsize=fontsize, ha="right", va="bottom")
    draw_text(fig, ax, 0.82, 0.83, names[2], colors[2], fontsize=fontsize, ha="left", va="bottom")
    draw_text(fig, ax, 0.87, 0.18, names[3], colors[3], fontsize=fontsize, ha="left", va="top")
    leg = ax.legend(names, loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.savefig(options.get("output", "./pdf/venn4.pdf"))

    return fig, ax


def venn5(labels, names=['A', 'B', 'C', 'D', 'E'], **options):
    """
    plots a 5-set Venn diagram

    @type labels: dict[str, str]
    @type names: list[str]
    @rtype: (Figure, AxesSubplot)

    input
      labels: a label dict where keys are identified via binary codes ('00001', '00010', '00100', ...),
              hence a valid set could look like: {'00001': 'text 1', '00010': 'text 2', '00100': 'text 3', ...}.
              unmentioned codes are considered as ''.
      names:  group names
      more:   colors, figsize, dpi, fontsize

    return
      pyplot Figure and AxesSubplot object
    """
    colors = options.get('colors', [default_colors[i] for i in range(5)])
    figsize = options.get('figsize', (16, 12))
    dpi = options.get('dpi', 96)
    fontsize = options.get('fontsize', 25)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(left=0.0, right=1.0)

    # body
    draw_ellipse(fig, ax, 0.428, 0.449, 0.87, 0.50, 155.0, colors[0])
    draw_ellipse(fig, ax, 0.469, 0.543, 0.87, 0.50, 82.0, colors[1])
    draw_ellipse(fig, ax, 0.558, 0.523, 0.87, 0.50, 10.0, colors[2])
    draw_ellipse(fig, ax, 0.578, 0.432, 0.87, 0.50, 118.0, colors[3])
    draw_ellipse(fig, ax, 0.489, 0.383, 0.87, 0.50, 46.0, colors[4])
    draw_text(fig, ax, 0.27, 0.11, labels.get('00001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.72, 0.11, labels.get('00010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.55, 0.13, labels.get('00011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.91, 0.58, labels.get('00100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.78, 0.64, labels.get('00101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.84, 0.41, labels.get('00110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.76, 0.55, labels.get('00111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.51, 0.9, labels.get('01000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.39, 0.15, labels.get('01001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.42, 0.78, labels.get('01010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.15, labels.get('01011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.67, 0.76, labels.get('01100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.70, 0.71, labels.get('01101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.51, 0.74, labels.get('01110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.64, 0.67, labels.get('01111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.10, 0.61, labels.get('10000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.20, 0.31, labels.get('10001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.76, 0.25, labels.get('10010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.65, 0.23, labels.get('10011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.18, 0.50, labels.get('10100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.21, 0.37, labels.get('10101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.81, 0.37, labels.get('10110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.74, 0.40, labels.get('10111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.27, 0.70, labels.get('11000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.34, 0.25, labels.get('11001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.33, 0.72, labels.get('11010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.51, 0.22, labels.get('11011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.25, 0.58, labels.get('11100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.28, 0.39, labels.get('11101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.36, 0.66, labels.get('11110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.51, 0.47, labels.get('11111', ''), fontsize=fontsize)

    # legend
    draw_text(fig, ax, 0.02, 0.72, names[0], colors[0], fontsize=fontsize, ha="right")
    draw_text(fig, ax, 0.78, 0.94, names[1], colors[1], fontsize=fontsize, va="bottom")
    draw_text(fig, ax, 0.97, 0.74, names[2], colors[2], fontsize=fontsize, ha="left")
    draw_text(fig, ax, 0.88, 0.05, names[3], colors[3], fontsize=fontsize, ha="left")
    draw_text(fig, ax, 0.12, 0.05, names[4], colors[4], fontsize=fontsize, ha="right")
    # leg = ax.legend(names, loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    plt.savefig(options.get("output", "./pdf/venn5.pdf"), bbox_inches='tight')

    return fig, ax


def venn6(labels, names=['A', 'B', 'C', 'D', 'E', 'F'], **options):
    """
    plots a 6-set Venn diagram

    @type labels: dict[str, str]
    @type names: list[str]
    @rtype: (Figure, AxesSubplot)

    input
      labels: a label dict where keys are identified via binary codes ('000001', '000010', '000100', ...),
              hence a valid set could look like: {'000001': 'text 1', '000010': 'text 2', '000100': 'text 3', ...}.
              unmentioned codes are considered as ''.
      names:  group names
      more:   colors, figsize, dpi, fontsize

    return
      pyplot Figure and AxesSubplot object
    """
    colors = options.get('colors', [default_colors[i] for i in range(6)])
    figsize = options.get('figsize', (40, 40))
    dpi = options.get('dpi', 96)
    fontsize = options.get('fontsize', 14)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.230, top=0.845)
    ax.set_xlim(left=0.173, right=0.788)

    # body
    # See https://web.archive.org/web/20040819232503/http://www.hpl.hp.com/techreports/2000/HPL-2000-73.pdf
    draw_triangle(fig, ax, 0.637, 0.921, 0.649, 0.274, 0.188, 0.667, colors[0])
    draw_triangle(fig, ax, 0.981, 0.769, 0.335, 0.191, 0.393, 0.671, colors[1])
    draw_triangle(fig, ax, 0.941, 0.397, 0.292, 0.475, 0.456, 0.747, colors[2])
    draw_triangle(fig, ax, 0.662, 0.119, 0.316, 0.548, 0.662, 0.700, colors[3])
    draw_triangle(fig, ax, 0.309, 0.081, 0.374, 0.718, 0.681, 0.488, colors[4])
    draw_triangle(fig, ax, 0.016, 0.626, 0.726, 0.687, 0.522, 0.327, colors[5])
    draw_text(fig, ax, 0.212, 0.562, labels.get('000001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.430, 0.249, labels.get('000010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.356, 0.444, labels.get('000011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.609, 0.255, labels.get('000100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.323, 0.546, labels.get('000101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.513, 0.316, labels.get('000110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.523, 0.348, labels.get('000111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.747, 0.458, labels.get('001000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.325, 0.492, labels.get('001001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.670, 0.481, labels.get('001010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.359, 0.478, labels.get('001011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.653, 0.444, labels.get('001100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.344, 0.526, labels.get('001101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.653, 0.466, labels.get('001110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.363, 0.503, labels.get('001111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.750, 0.616, labels.get('010000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.682, 0.654, labels.get('010001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.402, 0.310, labels.get('010010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.392, 0.421, labels.get('010011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.653, 0.691, labels.get('010100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.651, 0.644, labels.get('010101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.490, 0.340, labels.get('010110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.468, 0.399, labels.get('010111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.692, 0.545, labels.get('011000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.666, 0.592, labels.get('011001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.665, 0.496, labels.get('011010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.374, 0.470, labels.get('011011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.653, 0.537, labels.get('011100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.652, 0.579, labels.get('011101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.653, 0.488, labels.get('011110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.389, 0.486, labels.get('011111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.553, 0.806, labels.get('100000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.313, 0.604, labels.get('100001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.388, 0.694, labels.get('100010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.375, 0.633, labels.get('100011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.605, 0.359, labels.get('100100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.334, 0.555, labels.get('100101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.582, 0.397, labels.get('100110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.542, 0.372, labels.get('100111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.468, 0.708, labels.get('101000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.355, 0.572, labels.get('101001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.420, 0.679, labels.get('101010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.375, 0.597, labels.get('101011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.641, 0.436, labels.get('101100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.348, 0.538, labels.get('101101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.635, 0.453, labels.get('101110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.370, 0.548, labels.get('101111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.594, 0.689, labels.get('110000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.579, 0.670, labels.get('110001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.398, 0.670, labels.get('110010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.395, 0.653, labels.get('110011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.633, 0.682, labels.get('110100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.616, 0.656, labels.get('110101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.587, 0.427, labels.get('110110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.526, 0.415, labels.get('110111', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.495, 0.677, labels.get('111000', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.505, 0.648, labels.get('111001', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.428, 0.663, labels.get('111010', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.430, 0.631, labels.get('111011', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.639, 0.524, labels.get('111100', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.591, 0.604, labels.get('111101', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.622, 0.477, labels.get('111110', ''), fontsize=fontsize)
    draw_text(fig, ax, 0.501, 0.523, labels.get('111111', ''), fontsize=fontsize)

    # legend
    draw_text(fig, ax, 0.674, 0.824, names[0], colors[0], fontsize=fontsize)
    draw_text(fig, ax, 0.747, 0.751, names[1], colors[1], fontsize=fontsize)
    draw_text(fig, ax, 0.739, 0.396, names[2], colors[2], fontsize=fontsize)
    draw_text(fig, ax, 0.700, 0.247, names[3], colors[3], fontsize=fontsize)
    draw_text(fig, ax, 0.291, 0.255, names[4], colors[4], fontsize=fontsize)
    draw_text(fig, ax, 0.203, 0.484, names[5], colors[5], fontsize=fontsize)
    leg = ax.legend(names, loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.savefig(options.get("output", "./pdf/venn6.pdf"))

    return fig, ax


##%%


def calculate_intersections(sets):
    """
    Calculate the intersections of multiple sets and return a dictionary with binary labels.

    @type sets: list[set]
    @rtype: dict[str, set]

    input
      sets: a list of sets to calculate intersections for

    return
      label_dict: a dictionary where keys are binary labels and values are the corresponding set intersections
    """
    # 创建一个标签字典
    label_dict = {}

    def generate_label(n):
        if n <= 0:
            return []
        if n == 1:
            return ['0', '1']
        result = []
        for t in generate_label(n - 1):
            result.append(t + '0')
            result.append((t + '1'))
        return result

    labels = generate_label(len(sets))

    def get_intersection_by_label(label):
        target, rest = None, None
        for i in range(len(label)):
            if label[i] == '1':
                if target is None:
                    target = sets[i]
                else:
                    target = target.intersection(sets[i])
            else:
                if rest is None:
                    rest = sets[i]
                else:
                    rest = rest.union(sets[i])
        if target is None:
            return set()
        if rest is None:
            return target

        return target.difference(rest)

    for l in labels:
        label_dict[l] = get_intersection_by_label(l)

    return label_dict


def read_pass_result(file_path):
    result = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            if json_line['eval'] == 'PASS':
                result.add(json_line['project'] + "_" + json_line["bug_id"])
    return result


def read_all_result(file_path):
    result = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            result.add(json_line['project'] + "_" + json_line["bug_id"])
    return result


def read_pass_result_from_manual_file(file_path):
    result = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.split(",")[1].strip() == "True":
                result.add(line.split(",")[0].strip())
    return result


defects4j_history_bugs = get_history_defects4j_project_and_bug("/Users/tom/Downloads/defects4j-1.2.0")


def read_chatrepair_bugfix(file="./chatrepair/patch_all.txt"):
    result = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            result.append(line.strip().replace("-", "_"))
    return result


def read_chatrepair_bugfix_d4j12_and_d4j2(file="./chatrepair/patch_all.txt"):
    result = read_chatrepair_bugfix(file)
    result1, result2 = [], []
    for t in result:
        proj, bug_id = t.split("_")[0], t.split("_")[1]
        if bug_id in defects4j_history_bugs.get(proj, []):
            result1.append(t)
        else:
            result2.append(t)
    return result1, result2


def read_iter_bugfix_d4j12_and_d4j2(iter_dir='/Users/tom/Downloads/ITER-master'):
    correct_patch_dir = f'{iter_dir}/patches/correct'
    plausible_patch_dir = f'{iter_dir}/patches/plausible'
    correct_patches = os.listdir(correct_patch_dir)
    plausible_patches = os.listdir(plausible_patch_dir)
    correct_d4j12, correct_d4j2 = [], []
    plausible_d4j12, plausible_d4j2 = [], []

    d4j12_bugs = []
    for k in defects4j_history_bugs:
        for bid in defects4j_history_bugs[k]:
            d4j12_bugs.append(k + str(bid))
    for correct_patch in correct_patches:
        if correct_patch in d4j12_bugs:
            correct_d4j12.append(correct_patch)
        else:
            correct_d4j2.append(correct_patch)
    for plausible_patch in plausible_patches:
        if plausible_patch in d4j12_bugs:
            plausible_d4j12.append(plausible_patch)
        else:
            plausible_d4j2.append(plausible_patch)

    print(len(correct_d4j12), len(correct_d4j2), len(plausible_d4j12), len(plausible_d4j2))


def read_repairagent_bugfix(file="./repairagent/patch_list.txt"):
    result = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            result.append(line.strip().replace(" ", "_"))
    return result


def read_repairagent_bugfix_d4j12_and_d4j2(file="./repairagent/patch_list.txt"):
    result = read_repairagent_bugfix(file)
    result1, result2 = [], []
    for t in result:
        proj, bug_id = t.split("_")[0], t.split("_")[1]
        if bug_id in defects4j_history_bugs.get(proj, []):
            result1.append(t)
        else:
            result2.append(t)
    return result1, result2


##%%
repairagent_all = read_repairagent_bugfix()
repairagent_d4j12, repairagent_d4j2 = read_repairagent_bugfix_d4j12_and_d4j2()
chatrepair_all = read_chatrepair_bugfix()
chatrepair_d4j12, chatrepair_d4j2 = read_chatrepair_bugfix_d4j12_and_d4j2()
mcts_defects4j12 = read_manual_d4j12()
mcts_defects4j2 = read_manual_d4j2()
mcts_all = mcts_defects4j12.union(mcts_defects4j2)
cure_defects4j12 = ['Chart_11', 'Closure_62', 'Lang_59', 'Chart_12', 'Lang_6', 'Closure_126', 'Chart_17', 'Closure_70',
                    'Chart_14', 'Chart_1', 'Closure_73', 'Mockito_5', 'Math_98', 'Time_19', 'Math_58', 'Lang_38',
                    'Lang_10', 'Math_70', 'Mockito_29', 'Math_65', 'Math_59', 'Math_75', 'Lang_29', 'Mockito_38',
                    'Closure_11', 'Closure_10', 'Closure_38', 'Lang_26', 'Math_79', 'Math_50', 'Math_2', 'Math_82',
                    'Math_41', 'Math_57', 'Closure_18', 'Math_94', 'Math_80', 'Math_56', 'Lang_51', 'Mockito_8',
                    'Chart_24', 'Math_30', 'Closure_57', 'Chart_26', 'Math_27', 'Closure_40', 'Lang_43', 'Lang_57',
                    'Closure_92', 'Closure_86', 'Closure_102', 'Math_22', 'Math_34', 'Chart_9', 'Chart_8', 'Chart_20',
                    'Closure_46']

cure_defects4j2 = ['JxPath_5', 'JacksonDatabind_37', 'Jsoup_77', 'Jsoup_88', 'Collection_26', 'JacksonCore_25',
                   'Codec_17', 'Closure_168', 'JxPath_10', 'Cli_25', 'Cli_8', 'JacksonCore_5', 'Jsoup_68', 'Codec_4',
                   'JacksonDatabind_16', 'Jsoup_43', 'Codec_7', 'Codec_2', 'Compress_31']

rapgen_defects4j12 = ['Chart_1', 'Chart_11', 'Chart_12', 'Chart_16', 'Chart_20', 'Chart_7', 'Chart_8', 'Chart_9',
                      'Closure_104', 'Closure_106', 'Closure_11', 'Closure_113', 'Closure_115', 'Closure_123',
                      'Closure_125', 'Closure_126', 'Closure_18', 'Closure_19', 'Closure_46', 'Closure_57',
                      'Closure_65', 'Closure_70', 'Closure_73', 'Closure_75', 'Closure_77', 'Closure_79', 'Closure_86',
                      'Closure_92', 'Lang_10', 'Lang_26', 'Lang_29', 'Lang_34', 'Lang_35', 'Lang_51', 'Lang_52',
                      'Lang_57', 'Lang_6', 'Lang_60', 'Lang_7', 'Math_104', 'Math_22', 'Math_30', 'Math_34', 'Math_35',
                      'Math_41', 'Math_49', 'Math_5', 'Math_57', 'Math_67', 'Math_70', 'Math_72', 'Math_75', 'Math_77',
                      'Math_79', 'Math_80', 'Math_98', 'Mockito_26', 'Mockito_5', 'Time_19']
rapgen_defects4j2 = ['Cli_17', 'Cli_27', 'Cli_28', 'Cli_32', 'Cli_34', 'Cli_8', 'Closure_150', 'Closure_168',
                     'Codec_17', 'Codec_4', 'Codec_7', 'Codec_8', 'Collections_26', 'Compress_14', 'Compress_19',
                     'Compress_27', 'Compress_31', 'Compress_32', 'Csv_11', 'Csv_15', 'JacksonCore_14',
                     'JacksonCore_25', 'JacksonCore_5', 'JacksonCore_8', 'JacksonDatabind_102', 'JacksonDatabind_107',
                     'JacksonDatabind_13', 'JacksonDatabind_17', 'JacksonDatabind_27', 'JacksonDatabind_34',
                     'JacksonDatabind_46', 'JacksonDatabind_49', 'JacksonDatabind_54', 'JacksonDatabind_83',
                     'JacksonDatabind_99', 'JacksonXml_5', 'Jsoup_17', 'Jsoup_32', 'Jsoup_40', 'Jsoup_41', 'Jsoup_43',
                     'Jsoup_45', 'Jsoup_57', 'Jsoup_62', 'Jsoup_68', 'Jsoup_86', 'JxPath_5']

rewardrepair_defects4j12 = ['Math_57', 'Math_41', 'Time_19', 'Math_30', 'Math_82', 'Lang_29', 'Math_59', 'Math_34',
                            'Lang_45', 'Math_70', 'Closure_101', 'Chart_12', 'Lang_21', 'Lang_59', 'Closure_73',
                            'Math_85', 'Closure_62', 'Chart_24', 'Closure_86', 'Closure_70', 'Closure_18', 'Chart_11',
                            'Math_11', 'Lang_6', 'Closure_31', 'Lang_33', 'Mockito_26', 'Math_33', 'Math_75', 'Math_94',
                            'Math_80', 'Mockito_5', 'Closure_11', 'Chart_1', 'Closure_1', 'Math_50', 'Closure_126',
                            'Closure_92', 'Mockito_38', 'Chart_9', 'Closure_38', 'Math_104', 'Lang_57', 'Math_105',
                            'Math_2']
rewardrepair_defects4j2 = ['Closure_168', 'JxPath_16', 'JacksonDatabind_57', 'Jsoup_52', 'Jsoup_57',
                           'JacksonDatabind_27', 'Jsoup_64', 'JacksonCore_5', 'Csv_11', 'Codec_1', 'JacksonCore_25',
                           'Cli_25', 'JacksonDatabind_102', 'Csv_9', 'Codec_8', 'JxPath_10', 'Cli_8', 'Compress_14',
                           'JacksonDatabind_49', 'Jsoup_24', 'Cli_5', 'JacksonDatabind_17', 'Jsoup_43', 'Codec_7',
                           'JxPath_5', 'Jsoup_55', 'Codec_2', 'Codec_17', 'Cli_27', 'Codec_3', 'Cli_28', 'Jsoup_49',
                           'Compress_27', 'JacksonDatabind_13', 'JacksonDatabind_99', 'Cli_17', 'JxPath_1',
                           'Compress_31', 'Gson_6', 'Compress_19', 'JacksonDatabind_24', 'JacksonDatabind_83',
                           'Collections_26', 'JacksonDatabind_46', 'Jsoup_86']
selfapr_defects4j12 = ['Time_15', 'Lang_6', 'Lang_26', 'Lang_7', 'Math_11', 'Closure_113', 'Closure_73', 'Closure_79',
                       'Closure_109', 'Math_77', 'Math_80', 'Lang_34', 'Closure_6', 'Closure_104', 'Math_79',
                       'Closure_57', 'Mockito_26', 'Chart_1', 'Closure_75', 'Math_46', 'Closure_86', 'Lang_29',
                       'Mockito_5', 'Chart_13', 'Closure_46', 'Math_104', 'Math_82', 'Closure_106', 'Math_32',
                       'Closure_115', 'Closure_11', 'Lang_60', 'Math_30', 'Math_85', 'Math_41', 'Math_75', 'Chart_24',
                       'Closure_31', 'Lang_33', 'Closure_92', 'Chart_20', 'Closure_125', 'Math_72', 'Math_70',
                       'Math_73', 'Time_19', 'Time_7', 'Chart_8', 'Lang_10', 'Closure_38', 'Chart_9', 'Math_63',
                       'Chart_5', 'Closure_70', 'Closure_10', 'Math_94', 'Math_95', 'Math_5', 'Math_49', 'Closure_123',
                       'Math_59', 'Math_57', 'Lang_58', 'Closure_126', 'Mockito_29', 'Lang_21', 'Closure_62', 'Math_50',
                       'Chart_11', 'Lang_51', 'Lang_57', 'Closure_40', 'Chart_7', 'Math_22']
selfapr_defects4j2 = ['Compress_31', 'Cli_8', 'JacksonDatabind_27', 'Jsoup_17', 'Codec_7', 'Compress_18', 'Jsoup_24',
                      'Compress_14', 'JacksonDatabind_34', 'Jsoup_40', 'Cli_25', 'Codec_9', 'JacksonCore_5',
                      'Closure_168', 'Compress_38', 'Jsoup_46', 'JacksonDatabind_83', 'Collections_26', 'Codec_3',
                      'Cli_40', 'Cli_37', 'Cli_12', 'JacksonDatabind_12', 'Jsoup_68', 'Jsoup_41', 'Codec_4',
                      'JacksonCore_8', 'Cli_11', 'JxPath_12', 'JacksonDatabind_17', 'Csv_4', 'JacksonDatabind_99',
                      'JacksonDatabind_16', 'JacksonDatabind_46', 'Codec_8', 'JacksonCore_25', 'Compress_27',
                      'Jsoup_45', 'Jsoup_62', 'JacksonDatabind_57', 'Compress_19', 'Cli_17', 'JacksonDatabind_102',
                      'Gson_6', 'Compress_23', 'Codec_17', 'Codec_16']
# intersections = calculate_intersections([
#     set(mcts_defects4j12), set(rapgen_defects4j12), set(rewardrepair_defects4j12), set(selfapr_defects4j12),
#     set(cure_defects4j12)
# ])
# intersections = {k: len(v) for k, v in intersections.items()}
# intersections2 = calculate_intersections([
#     set(mcts_defects4j2), set(rapgen_defects4j2), set(rewardrepair_defects4j2), set(selfapr_defects4j2),
#     set(cure_defects4j2)
# ])
# intersections2 = {k: len(v) for k, v in intersections2.items()}
# intersections3 = calculate_intersections([
#     set(rapgen_defects4j12), set(rewardrepair_defects4j12), set(selfapr_defects4j12), set(cure_defects4j12)
# ])
# intersections3 = {k: len(v) for k, v in intersections3.items()}
# intersections4 = calculate_intersections([
#     set(rapgen_defects4j2), set(rewardrepair_defects4j2), set(selfapr_defects4j2), set(cure_defects4j2)
# ])
# intersections4 = {k: len(v) for k, v in intersections4.items()}
# print(intersections3)
# venn4(intersections3,names=['RAP-Gen','RewardRepair','SelfAPR','CURE'],output="./pdf/d4j12_venn4")
# venn5(intersections, names=['APRMCTS', 'RAP-Gen', 'RewardRepair', 'SelfAPR', 'CURE'], output="./pdf/d4j12_venn.pdf")
# venn5(intersections,names=['APRMCTS','RAP-Gen','RewardRepair','SelfAPR','CURE'],output="./pdf/d4j12_venn.pdf")
##%%

# intersections5=calculate_intersections([
#     set(repairagent_all),set(mcts_all),set(chatrepair_all)
# ])
# intersections5 = {k: len(v) for k, v in intersections5.items()}
# venn3(intersections5,names=['RepairAgent','APRMCTS','ChatRepair'],output="./pdf/d4jall_venn.pdf")

# intersections6 = calculate_intersections([
#     set(mcts_defects4j12), set(repairagent_d4j12), set(chatrepair_d4j12)
# ])
#
# intersections7 = calculate_intersections([
#     set(mcts_defects4j2), set(repairagent_d4j2), set(chatrepair_d4j2)
# ])
# intersections6 = {k: len(v) for k, v in intersections6.items()}
# intersections7 = {k: len(v) for k, v in intersections7.items()}
# venn3(intersections6, names=['APRMCTS', 'RepairAgent', 'ChatRepair'], output="./pdf/best_d4j12_venn.pdf")

read_iter_bugfix_d4j12_and_d4j2()
