python3 feature_extractor.py ../../labAHLT/data/test > feats_test.dat
python3 feature_extractor.py ../../labAHLT/data/train > feats.dat
python3 learner.py feats.dat feats_test.dat output.dat
python3 evaluator.py DDI ../../labAHLT/data/test output.dat