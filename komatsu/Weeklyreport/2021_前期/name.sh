lualatex $1.tex
 mv $1.pdf $2.pdf
#evince $2.pdf 
rm $1.aux
rm $1.log
