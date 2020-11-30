#!/bin/bash

[ $# == 1 ] || exit 1

fname="${1%.*}"

pygmentize -O style=solarized-light,linenos=1,font_size=20,font_name='Menlo' -o "${fname}.png" "$1"
