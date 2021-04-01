
for c1 in $(seq 0 0.1 0.3)
do
    for c2 in $(seq 0 0.001 0.01)
    do
        python3 crf-learner.py model features_train.txt $c1 $c2
        python3 crf-classifier.py model features_devel.txt > devel_classified.out
        python3 evaluator-test.py NER ../labAHLT/data/devel/ devel_classified.out
    done
done
