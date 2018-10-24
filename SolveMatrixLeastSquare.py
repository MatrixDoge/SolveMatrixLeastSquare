import numpy
import fractions
numpy.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
 
def get_least_square(mInput, mResult):
    mInputTranspose = numpy.transpose(mInput)
    mAtA = numpy.matmul(mInputTranspose, mInput)
    mAtAInv = numpy.linalg.inv(mAtA)
    mAtR = numpy.matmul(mInputTranspose, mResult)
    mBetaHat = numpy.matmul(mAtAInv, mAtR)
    return mBetaHat



a = numpy.array([[2, 1], [1, -1], [1, 1]])
r = numpy.array([[3], [1], [2]])
print(get_least_square(a,r))