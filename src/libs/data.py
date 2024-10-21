import numpy as np
import pandas
from scipy.spatial.transform import Rotation as R


def read_frames(logfile: str) -> np.ndarray:
    """
    Read frames from a log file.

    :param logfile: The data file to read from.
    :return: Array containing t, x, y, z, qx, qy, qz, qw.
    """

    data = pandas.read_csv(logfile, sep="\t", engine="python")

    # split data
    t = data.TIME.to_numpy().reshape((-1, 1))
    position = np.vstack((data.X.to_numpy(), data.Y.to_numpy(), data.Z.to_numpy())).T * 0.001
    orientation = np.vstack((data.A.to_numpy(), data.B.to_numpy(), data.C.to_numpy())).T
    orientation = R.from_euler("ZYX", orientation, degrees=True).as_quat()

    return np.hstack((t, position, orientation))
