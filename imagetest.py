import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from pathlib import Path
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

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


def gif(filename, array, fps=1000, scale=1.0):
    """Creates a gif given a stack of images using moviepy

    Code from: https://gist.github.com/nirum/d4224ad3cd0d71bfef6eba8f3d6ffd59
    Credit to Niru Maheswaranatha
    
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps, program='ffmpeg', opt='optimizeplus')
    return clip



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

   frames = []
   waves = []
   waves_and_frames = []

   # only process every other index because every frequency has the
   # same magnitude as its symmetric opposite about the origin.

   pbar = tqdm(total=len(decreasingIndices[::2]))

   for i,j in decreasingIndices[::2]:
      wave = np.zeros(A.shape)

      entry = ffted[i,j]
      frame[i, j] = wave[i, j] = entry
      frame[-i, -j] = wave[-i, -j] = entry.conjugate()

      ifftFrame = np.fft.ifft2(np.copy(frame))
      ifftFrame = [[x.real for x in row] for row in ifftFrame]
      # imageio.imwrite('frames/%06d.png' % t, np.array(ifftFrame, dtype=np.uint8))
      frames.append(np.array(ifftFrame, dtype=np.uint8))

      ifftWave = np.fft.ifft2(np.copy(wave))
      ifftWave = [[x.real for x in row] for row in ifftWave]
      # imageio.imwrite('waves/%06d.png' % t, np.array(ifftWave, dtype=np.uint8))
      waves.append(np.array(ifftWave, dtype=np.uint8))

      waves_and_frames.append(np.hstack((np.array(ifftWave, dtype=np.uint8), np.array(ifftFrame, dtype=np.uint8))))

      t += 1

      pbar.update(1)
   
   pbar.close()

   gif('images/frames', np.array(frames))
   gif('images/waves', np.array(waves))

   gif('images/waves_and_frames', np.array(waves_and_frames))


#ifftExample()
# fourierSpectrumExample('images/sherlock.jpg')
animation('images/hance-up-sw.png')
