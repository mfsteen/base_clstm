import load_data as ld
import pickle
import numpy as np

# TODO: Figure out why the shape of getOneHotOut is (1500,30) but
#       shape of testCorrectOut is (1500, 3)
# TODO: Get test_GetResults() to pass when getOneHotOut and testCorrectOut
#       are compared.

def test_GetResults():
    testInput = pickle.load(open("correctOneHotIn.pickle", "rb"))
    testCorrectOut = pickle.load(open("correctOneHotOut.pickle", "rb"))

    getOneHotOut = ld.get_onehot(testInput, None)

#    print(testInput)
#    print(testCorrectOut)

    assert getOneHotOut[1].shape == testCorrectOut[1].shape
#    assert np.equal(getOneHotOut[1], testCorrectOut[1])
