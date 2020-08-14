
import sys
sys.path.insert(1, "../code_development/")
import load_data as ld

import pickle
import numpy as np

def test_GetResults():
    testInput = pickle.load(open("data_testing/correctOneHotIn.pickle", "rb"))
    testCorrectOut = pickle.load(open("data_testing/correctOneHotOut.pickle", "rb"))

    getOneHotOut = ld.get_onehot(testInput, None, num_classes=3, seq_len=20)

#    print(testInput)
#    print(testCorrectOut)

    for x,y in zip(testCorrectOut,getOneHotOut):
        assert np.equal(x,y).all()
    #assert getOneHotOut[1].shape == testCorrectOut[1].shape
    #assert np.equal(getOneHotOut[1], testCorrectOut[1])
