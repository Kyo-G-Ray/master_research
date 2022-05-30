#!/bin/bash
platex main.tex
pbibtex main
platex main.tex
dvipdfmx main.dvi

#正しいやつ

#/chess/project/project1/shell/check2.sh

evince main.pdf &

#coverを出力するやつ

!/bin/bash
 platex cover2.tex
 platex cover2.tex
 pbibtex cover2
 dvipdfmx cover2.dvi

# /chess/project/project1/shell/check2.sh
# evince cover2.pdf &

