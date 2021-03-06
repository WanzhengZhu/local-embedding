#!/bin/bash
corpusName=${1:-toy}    ## Name of the input corpus
taxonName=non-para      ## Name of the taxonomy
FIRST_RUN=${FIRST_RUN:- 0}      ## If need preprocessing from raw input, set it to be 1, otherwise, set 0

if [ $FIRST_RUN -eq 1 ]; then
	echo 'Start data preprocessing'
	## compile word2vec for embedding learning
	gcc word2vec.c -o word2vec -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

	## create initial folder if not exist
	if [ ! -d ../data/$corpusName/init ]; then
		mkdir ../data/$corpusName/init
	fi

	echo 'Start cluster-preprocess.py'
	time python cluster-preprocess.py $corpusName

	echo 'Start preprocess.py'
	time python preprocess.py $corpusName

	cp ../data/$corpusName/input/embeddings.txt ../data/$corpusName/init/embeddings.txt
	cp ../data/$corpusName/input/keywords.txt ../data/$corpusName/init/seed_keywords.txt
fi

## create root folder for taxonomy
if [ ! -d ../data/$corpusName/$taxonName ]; then
	mkdir ../data/$corpusName/$taxonName
fi

echo 'Start TaxonGen'
python main.py -dataset $corpusName -foldername $taxonName

echo 'Generate compressed taxonomy'
if [ ! -d ../data/$corpusName/taxonomies ]; then
	mkdir ../data/$corpusName/taxonomies
fi

python compress_1.py -root ../data/$corpusName/$taxonName -output ../data/$corpusName/taxonomies/$taxonName-short.txt
python compress.py -root ../data/$corpusName/$taxonName -output ../data/$corpusName/taxonomies/$taxonName.txt
