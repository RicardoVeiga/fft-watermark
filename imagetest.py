import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

current_path = Path(__file__).parent.absolute()

def ifftExample():
   shift = 4
   spectrum = np.zeros((513,513))
   spectrum[256,256] = 1
   spectrum[256,256 + shift] = 1
   spectrum[256,256 - shift] = 1

   plt.clf()
   plt.imshow(spectrum, cmap=plt.cm.gray)
   plt.show()


   A = np.fft.ifft2(spectrum)
   img = np.array([[np.absolute(x) for x in row] for row in A])

   plt.clf()
   plt.imshow(img, cmap=plt.cm.gray)
   plt.show()


def fourierSpectrumExample(filename):
   A = imageio.imread(filename, as_gray=True)

   unshiftedfft = np.fft.fft2(A)
   spectrum = np.log10(np.absolute(unshiftedfft) + np.ones(A.shape))
   imageio.imwrite("%s-spectrum-unshifted.png" % (filename.split('.')[0]), np.array(spectrum, dtype=np.uint8))

   shiftedFFT = np.fft.fftshift(np.fft.fft2(A))
   spectrum = np.log10(np.absolute(shiftedFFT) + np.ones(A.shape))
   imageio.imwrite("%s-spectrum.png" % (filename.split('.')[0]), np.array(spectrum, np.uint8))


# create a list of 2d indices of A in decreasing order by the size of the
# (real) entry of A that they index to
def sortedIndices(A):
   indexList = [(i,j) for i in range(A.shape[0]) for j in range(A.shape[1])]
   indexList.sort(key=lambda x: -A[x])
   return np.array(indexList)


def animation(filename):
   A = imageio.imread(filename, as_gray=True)

   # subtract the mean so that the DC component is zero
   A = A - np.mean(A)
   ffted = np.fft.fft2(A)

   magnitudes = np.absolute(ffted)
   frame = np.zeros(ffted.shape, dtype=np.complex)

   t = 0
   decreasingIndices = sortedIndices(magnitudes)

   Path("{}/frames".format(current_path)).mkdir(parents=True, exist_ok=True)
   Path("{}/waves".format(current_path)).mkdir(parents=True, exist_ok=True)

   # only process every other index because every frequency has the
   # same magnitude as its symmetric opposite about the origin.
   for i,j in decreasingIndices[::2]:
      wave = np.zeros(A.shape)

      entry = ffted[i,j]
      frame[i, j] = wave[i, j] = entry
      frame[-i, -j] = wave[-i, -j] = entry.conjugate()

      ifftFrame = np.fft.ifft2(np.copy(frame))
      ifftFrame = [[x.real for x in row] for row in ifftFrame]
      imageio.imwrite('frames/%06d.png' % t, np.array(ifftFrame, dtype=np.uint8))

      ifftWave = np.fft.ifft2(np.copy(wave))
      ifftWave = [[x.real for x in row] for row in ifftWave]
      imageio.imwrite('waves/%06d.png' % t, np.array(ifftWave, dtype=np.uint8))

      t += 1


#ifftExample()
fourierSpectrumExample('images/sherlock.jpg')
animation('images/hance-up-sw.png')


